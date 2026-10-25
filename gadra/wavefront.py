"""Bounded attention-patch programs on the existing execution resources.

The host closure is a dependency proof, not a cache or a free hardware queue.
Runtime index reads, CSR construction and demand-bitmap accesses are timed.
FIFO patch pipelining is public; coverage can change its consumption order.
"""
from copy import copy
import math

import numpy as np

from .graph import key, rows
from .protocol import SECTOR_BYTES


def row_local(event):
    return (len(event['outputs']) == 1 and len(event['outputs'][0]['shape']) == 2
            and (event['kind'] == 'linear' or event.get('op', '').startswith((
                'aten.add.', 'aten.sub.', 'aten.mul.', 'aten.div.', 'aten.relu',
                'aten.gelu', 'aten.layer_norm', 'aten.to.', 'aten.clamp_min',
                'aten.linalg_vector_norm'))))


def regions(graph, boundaries):
    """Recognize QKV -> actual gather -> SDPA -> inverse -> row-local tail.

    Metadata, sparse operators, global reductions and observations remain
    boundaries. Both casted Chorus and uncasted SceneSplat retain their views.
    """
    found = {}
    for attention, event in enumerate(graph.events):
        if event['kind'] != 'sdpa' or len(event['outputs'][0]['shape']) != 4:
            continue
        before = attention - 1
        while before >= 0 and graph.events[before].get('op', '').startswith('aten.to.'):
            before -= 1
        after = attention + 1
        while after < len(graph.events) and graph.events[after].get('op', '').startswith('aten.to.'):
            after += 1
        if before < 1 or after >= len(graph.events):
            continue
        gather, inverse = graph.events[before], graph.events[after]
        if any(e.get('op') != 'aten.index.Tensor' for e in (gather, inverse)):
            continue
        start = before - 1
        producer = graph.events[start]
        if (producer['kind'] != 'linear' or len(producer['outputs'][0]['shape']) != 2
                or key(producer['outputs'][0]) != key(gather['inputs'][0])):
            continue
        n = rows(producer['outputs'][0])
        shape = event['outputs'][0]['shape']
        patches, _, sequence, _ = shape
        indices = graph.maps.get('indices', {})
        if len(indices.get(before, [])) != 1 or len(indices.get(after, [])) != 1:
            continue
        padding = np.asarray(indices[before][0]).reshape(-1)
        mapping = np.asarray(indices[after][0]).reshape(-1)
        if (len(mapping) != n or len(padding) != patches * sequence
                or np.any(mapping < 0) or np.any(mapping >= len(padding))
                or not np.array_equal(padding[mapping], np.arange(n))):
            continue
        end = after
        while end + 1 < len(graph.events) and end not in boundaries:
            following = graph.events[end + 1]
            if not row_local(following) or rows(following['outputs'][0]) != n:
                break
            end += 1
        if any(start <= at < end for at in boundaries):
            continue
        internal = {key(s) for e in graph.events[start:end + 1] for s in e['outputs']}
        external_views = {}
        for e in graph.events[start:end + 1]:
            for s in e['inputs']:
                if s['role'] != 'feature' or key(s) in internal:
                    continue
                physical = graph.bases[key(s)]
                signature = (s['address'], tuple(s['shape']), tuple(s['stride']), s['element_bytes'])
                external_views.setdefault(physical, set()).add(signature)
        if any(len(views) > 1 for views in external_views.values()):
            # Per-row last-use reclamation is only local to one event. A
            # shifted external view could still read a retired physical row
            # in a later patch. Keep the original schedule in this case.
            continue
        # No writes/carry or hidden global operations are admitted by this
        # structural recognizer. Exact byte-view closure is checked below.
        found[start] = dict(start=start, end=end, attention=attention,
                            inverse=after, mapping=mapping // sequence,
                            patches=patches)
    return found


def region_graph(graph, region):
    local = copy(graph)
    start, end = region['start'], region['end']
    local.events = graph.events[start:end + 1]
    local.maps = dict(graph.maps, checkpoints=[], indices={
        at - start: value for at, value in graph.maps.get('indices', {}).items()
        if start <= at <= end})
    local.demand = {}
    return local


def plans(graph, region, live, order):
    """Yield once-only rows; keep full K/V supply for padding/partial outputs."""
    local = region_graph(graph, region)
    start, end = region['start'], region['end']
    seen = [np.empty(0, np.int64) for _ in local.events]
    terminal = graph.events[end]['outputs'][0]
    for patch in order:
        selected = live[end][region['mapping'][live[end]] == patch]
        required = local.closure([(terminal, selected)])
        if required.carry or local.conservative:
            raise ValueError('Attention wavefront requires exact immutable dependencies')
        incremental = []
        for offset, wanted in enumerate(required):
            wanted = np.intersect1d(wanted, live[start + offset], assume_unique=True)
            incremental.append(np.setdiff1d(wanted, seen[offset], assume_unique=True))
            seen[offset] = np.union1d(seen[offset], wanted)
        yield int(patch), required, incremental
    if any(not np.array_equal(done, live[start + offset]) for offset, done in enumerate(seen)):
        raise ValueError('Terminal demand does not span the complete attention region')


def prepare(sim, region):
    """Reuse the timed inverse-row CSR program and its finite scratch owner."""
    from .feeder import inverse_row_scratch
    mapping = region['mapping']
    n = region['patches']
    layout = inverse_row_scratch(len(mapping), n)
    if layout[-1] > sim.output_budget():
        return None
    offsets = np.r_[0, np.cumsum(np.bincount(mapping, minlength=n))]
    plan = dict(mapping=mapping, consumer_ids=None, layout=layout, offsets=offsets,
                order=np.argsort(mapping, kind='stable'), bytes=(len(mapping) + n + 1) * 4,
                scratch=layout[-1], prepared=False)
    plan['base'] = sim.allocate_workspace(plan['bytes'])
    # The inverse values must arrive before integer division creates keys.
    ready = max(sim.cycle, sim.memory.clock)
    for spec in sim.g.events[region['inverse']]['inputs'][1:]:
        if spec['role'] == 'index':
            ready = max(ready, sim.memory.access(sim.g.sectors(spec, np.arange(rows(spec)))))
    arithmetic = math.ceil(len(mapping) / sim.c['vector_lanes'])
    sim.cycle = sim.issue_vector(ready, arithmetic)
    sim.stats['vector_cycles'] += arithmetic
    sim.memory.clock = max(sim.memory.clock, sim.cycle)
    sim.prepare_final_gather(region['inverse'], plan)
    sim.stats['patch_csr_bytes'] += plan['bytes']
    return plan


def charge_demands(sim, region, required):
    """Replay map traversal and finite control-bank read/OR/write work.

    Two demand bitmaps per event are reserved from control SRAM by schedule:
    current request and generated rows. No per-leaf task list is held there.
    Each lane batch resolves actual word/bank conflicts before its update.
    """
    ports, lanes = sim.c['cam_lookups_per_cycle'], sim.c['vector_lanes']
    cost = arithmetic = transferred = 0
    for offset, ids in enumerate(required):
        event = sim.g.events[region['start'] + offset]
        if event.get('op') == 'aten.index.Tensor':
            for spec, selected in sim.g.inputs_for(region['start'] + offset, ids):
                if spec['role'] == 'index':
                    sim.memory.clock = max(sim.memory.clock, sim.memory.access(sim.g.sectors(spec, selected)))
        for begin in range(0, len(ids), ports):
            batch = ids[begin:begin + ports]
            rounds = int(np.bincount((batch // 32) % ports, minlength=ports).max(initial=0))
            # IDs, old generated bits, and current demand; one OR and test.
            work = 2 * math.ceil(len(batch) / lanes)
            cost += 4 * rounds + work
            arithmetic += work
            transferred += (len(batch) + 4 * rounds) * 4
    done = sim.issue_vector(max(sim.cycle, sim.memory.clock), cost)
    sim.cycle = done
    sim.memory.clock = max(sim.memory.clock, done)
    sim.stats['vector_cycles'] += arithmetic
    sim.stats['patch_demand_cycles'] += cost
    sim.stats['patch_control_bytes'] += transferred


def patch_order(sim, region, terminal):
    """Generate a finite descriptor queue; B reads actual coverage row IDs."""
    mapping = region['mapping']
    active = np.unique(mapping[terminal])
    if not sim.b:
        order = active
        cost = math.ceil(len(order) * 2 / sim.c['cam_lookups_per_cycle'])
    else:
        ordered = sim.row_order(region['end'], np.arange(len(mapping)))
        _, first = np.unique(mapping[ordered], return_index=True)
        all_patches = mapping[ordered][np.sort(first)]
        order = all_patches[np.isin(all_patches, active)]
        # Coverage construction supplies the permutation in external metadata;
        # its runtime read is not replaced by the host ndarray.
        base = sim.allocate_workspace(len(ordered) * 4)
        ready = sim.memory.access(np.arange(base // SECTOR_BYTES,
            math.ceil((base + len(ordered) * 4) / SECTOR_BYTES)))
        for spec in sim.g.events[region['inverse']]['inputs'][1:]:
            if spec['role'] == 'index':
                ready = max(ready, sim.memory.access(sim.g.sectors(spec, np.arange(rows(spec)))))
        sim.memory.clock = max(sim.memory.clock, ready)
        cost = 0
        ports = sim.c['cam_lookups_per_cycle']
        seen = set()
        for begin in range(0, len(ordered), ports):
            keys = mapping[ordered[begin:begin + ports]]
            words = np.unique(keys // 32)
            reads = int(np.bincount(words % ports, minlength=ports).max(initial=0))
            new = set(keys.tolist()) - seen
            seen.update(new)
            # Read seen bits, gather/divide/test, merge duplicate lane keys,
            # update seen words, and enqueue the new patch descriptors.
            merge = math.ceil(math.log2(max(1, len(keys))))
            arithmetic = 3 * math.ceil(len(keys) / sim.c['vector_lanes']) + merge
            cost += reads * (1 + bool(new)) + arithmetic + math.ceil(2 * len(new) / ports)
            sim.stats['vector_cycles'] += arithmetic
            sim.stats['patch_control_bytes'] += reads * 4 * (1 + bool(new)) + len(new) * 8
        inverse_bytes = sum(s['bytes'] for s in sim.g.events[region['inverse']]['inputs'][1:]
                            if s['role'] == 'index')
        sim.stats['patch_order_read_bytes'] += len(ordered) * 4 + inverse_bytes
    sim.cycle = sim.issue_vector(max(sim.cycle, sim.memory.clock), cost)
    sim.memory.clock = max(sim.memory.clock, sim.cycle)
    sim.stats['patch_order_cycles'] += cost
    return order


def schedule(sim, region, live, control):
    """Stream patches in one execute invocation; final flags are global."""
    start, end = region['start'], region['end']
    terminal = live[end]
    if not len(terminal) or region['patches'] < 2:
        return False
    # Only the numerical ordering is computed in the proof pass. The timed
    # program below builds/reads its finite queue before any feature issue.
    ordered = sim.row_order(end, np.arange(len(region['mapping'])))
    _, first = np.unique(region['mapping'][ordered], return_index=True)
    candidate = region['mapping'][ordered][np.sort(first)] if sim.b else np.arange(region['patches'])
    order = candidate[np.isin(candidate, np.unique(region['mapping'][terminal]))]
    state_bytes = 2 * sum(math.ceil(rows(e['outputs'][0]) / 32) * 4
                          for e in sim.g.events[start:end + 1])
    state_bytes += region['patches'] * 8 + math.ceil(region['patches'] / 32) * 4
    if state_bytes >= sim.control_capacity() or region['patches'] > sim.c['queue_entries']:
        return False
    # Compile-time check before changing any execution state. This is an
    # exact proof for the captured program, never evaluation-label selection.
    try:
        for _ in plans(sim.g, region, live, order):
            pass
    except ValueError:
        return False
    csr = prepare(sim, region)
    if csr is None:
        return False
    original_config = sim.c
    sim.c = dict(sim.c, control_sram_bytes=sim.c['control_sram_bytes'] - state_bytes)
    sim.stats['patch_state_peak_bytes'] = max(sim.stats['patch_state_peak_bytes'], state_bytes)
    clear = math.ceil(state_bytes / (4 * sim.c['cam_lookups_per_cycle']))
    sim.cycle = max(sim.cycle, sim.memory.clock) + clear
    sim.memory.clock = sim.cycle
    sim.stats['patch_control_bytes'] += state_bytes
    timed_order = patch_order(sim, region, terminal)
    np.testing.assert_array_equal(timed_order, order)
    filtered = len(terminal) != len(region['mapping'])
    if filtered:
        # Uses the terminal current-demand bitmap already reserved above.
        sim.initialize_gather_demand(terminal, len(region['mapping']))
    remaining = [len(live[at]) for at in range(start, end + 1)]
    try:
        for number, (patch, required, incremental) in enumerate(plans(sim.g, region, live, order)):
            links = np.arange(csr['offsets'][patch], csr['offsets'][patch + 1])
            addresses = np.r_[csr['base'] + np.array([patch, patch + 1]) * 4,
                              csr['base'] + len(csr['offsets']) * 4 + links * 4]
            sim.memory.clock = max(sim.memory.clock, sim.memory.access(np.unique(addresses // SECTOR_BYTES)))
            sim.memory.clock += math.ceil(2 / sim.c['cam_lookups_per_cycle'])
            sim.stats['patch_control_bytes'] += 8
            if filtered:
                consumers = csr['order'][links]
                words = np.unique(consumers // 32)
                ports = sim.c['cam_lookups_per_cycle']
                reads = int(np.bincount(words % ports, minlength=ports).max(initial=0))
                arithmetic = math.ceil(len(consumers) / sim.c['vector_lanes'])
                sim.cycle = sim.issue_vector(max(sim.cycle, sim.memory.clock),
                    reads + sim.c['cam_latency_cycles'] + arithmetic)
                sim.memory.clock = max(sim.memory.clock, sim.cycle)
                sim.stats['vector_cycles'] += arithmetic
                sim.stats['patch_activity_test_rows'] += len(consumers)
                sim.stats['patch_control_bytes'] += len(words) * 4
            charge_demands(sim, region, required)
            floor = sim.cycle
            partial = list(live)
            partial[start:end + 1] = incremental
            for at, event, ids, _ in sim.schedule(partial, control, start, end + 1, False):
                if not len(ids):
                    continue
                remaining[at - start] -= len(ids)
                last = remaining[at - start] == 0
                sim.chain_floor = floor if sim.c.get('row_chain_overlap', False) else None
                sim.chain_end = number == len(order) - 1 and at == end and last
                yield at, event, ids, last
            sim.stats['attention_patch_programs'] += 1
        if any(remaining):
            raise AssertionError('Patch program lost requested rows')
        sim.stats['attention_wavefront_regions'] += 1
    finally:
        sim.c = original_config
    return True
