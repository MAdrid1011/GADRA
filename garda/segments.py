"""Captured CSR reductions on the existing finite vector/output datapath.

Only axis-zero, two-dimensional max/min with real offsets and no
initial value are lowered. Other calls keep the graph's conservative path.
Offsets specify topology, not feature values or future early-stop decisions.
"""
import math

import numpy as np


# Additive reductions retain their original execution path: a parallel tree
# can change floating point results, including coordinates used for geometry.
REDUCTIONS = ('max', 'min')


def descriptor(graph, index):
    event = graph.events[index]
    if (event.get('op') != 'aten.segment_reduce.default'
        or event.get('segment_reduction') not in REDUCTIONS):
        return None
    # Region graphs share real events but remap their program counters. A
    # cache keyed only by the local PC can alias an unrelated main event.
    cache = graph.__dict__.setdefault('segment_descriptors', {})
    identity = id(event)
    if identity in cache:
        return cache[identity][1]
    source, pointer = event['inputs']
    output, = event['outputs']
    values, = graph.maps['indices'][index]
    offsets = np.asarray(values)
    if (event.get('segment_axis') != 0 or event['segment_reduction'] not in REDUCTIONS
        or offsets.dtype.kind not in 'iu' or offsets.ndim != 1
        or len(source['shape']) != 2 or len(output['shape']) != 2
        or source['shape'][1] != output['shape'][1]
        or pointer['shape'] != [output['shape'][0] + 1, 1]
        or len(offsets) != output['shape'][0] + 1
        or offsets[0] != 0 or offsets[-1] != source['shape'][0]
        or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError('Invalid captured axis-zero CSR reduction')
    cache[identity] = (event, offsets.astype(np.int64, copy=False))
    return cache[identity][1]


def source_rows(offsets, selected):
    """Expand disjoint selected spans without constructing an N-by-S mask."""
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    lengths = offsets[selected + 1] - offsets[selected]
    total = int(lengths.sum())
    if not total:
        return np.empty(0, dtype=np.int64)
    starts = np.cumsum(lengths) - lengths
    return np.arange(total, dtype=np.int64) + np.repeat(offsets[selected] - starts, lengths)


def group_rows(sim, index, selected):
    """Bound expanded FP32 staging by one existing physical output owner."""
    offsets = descriptor(sim.g, index)
    lengths = offsets[selected + 1] - offsets[selected]
    width = sim.g.events[index]['outputs'][0]['shape'][1]
    row_bytes = width * 4
    largest = max(1, int(lengths.max(initial=0)))
    count = sim.output_budget() // (largest * row_bytes)
    if not count:
        raise ValueError('One CSR segment exceeds the finite output owner')
    return count


def decode(sim, selected, ready):
    """Returned pointers must be decoded before issuing dependent loads."""
    cost = math.ceil(2 * len(selected) / sim.c['vector_lanes'])
    done = sim.issue_vector(ready, cost)
    sim.stats['vector_cycles'] += cost
    sim.stats['segment_vector_cycles'] += cost
    return done


def reduce(sim, index, selected, ready):
    """Reserve index decode, FP32 staging, tree passes and final row drain.

    Every level reads its current packed rows before overwriting the same
    owner. Independent windows may prefetch in the other existing owner;
    their arithmetic still shares the vector issue queue and bank calendars.
    """
    event = sim.g.events[index]
    source = event['inputs'][0]
    output = event['outputs'][0]
    offsets = descriptor(sim.g, index)
    lengths = offsets[selected + 1] - offsets[selected]
    width = output['shape'][1]
    lanes = sim.c['vector_lanes']
    actual_rows = int(lengths.sum())
    # An empty segment produces its defined infinite reduction identity; it
    # still needs a row in the finite tree and an output write.
    remaining = np.maximum(lengths, 1)
    staged_bytes = int(remaining.sum()) * width * 4
    if staged_bytes > sim.output_budget():
        raise AssertionError('Expanded CSR staging exceeds output SRAM')
    sim.stats['segment_peak_staging_bytes'] = max(sim.stats['segment_peak_staging_bytes'], staged_bytes)
    sim.stats['segment_source_rows'] += actual_rows
    sim.stats['segment_output_rows'] += len(selected)

    def vector(cost):
        nonlocal ready
        ready = sim.issue_vector(ready, cost)
        sim.stats['vector_cycles'] += cost
        sim.stats['segment_vector_cycles'] += cost

    if source['element_bytes'] != 4:
        vector(math.ceil(actual_rows * width / lanes))
    if np.any(lengths == 0):
        vector(math.ceil(int((lengths == 0).sum()) * width / lanes))
    ready = sim.output_transfer(staged_bytes, ready, write=True)
    while np.any(remaining > 1):
        ready = sim.output_transfer(int(remaining.sum()) * width * 4, ready)
        pairs = int((remaining // 2).sum())
        vector(math.ceil(pairs * width / lanes))
        sim.stats['segment_reduction_ops'] += pairs * width
        remaining = (remaining + 1) // 2
        ready = sim.output_transfer(int(remaining.sum()) * width * 4, ready, write=True)
    ready = sim.output_transfer(len(selected) * width * 4, ready)
    if output['element_bytes'] != 4:
        vector(math.ceil(len(selected) * width / lanes))
    return ready
