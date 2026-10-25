"""Fuse exact adjacent vector producers into one finite writeback program."""

from collections import defaultdict

from .graph import key, rows


def integer_mask_candidates(graph):
    """Find ``bitwise_and -> ne(0)`` pairs with no observable intermediate."""
    if not hasattr(graph, "trace"):
        return {}
    users = defaultdict(list)
    writers = defaultdict(list)
    for at, event in enumerate(graph.events):
        for spec in event["inputs"]:
            users[key(spec)].append(at)
        for spec in event["outputs"]:
            writers[key(spec)].append(at)
    observed = {
        key(spec)
        for checkpoint in graph.maps.get("checkpoints", [])
        for spec in ([checkpoint["spec"]] if "spec" in checkpoint else [])
        + checkpoint.get("readout_inputs", [])
    }
    observed.update(key(spec) for spec in graph.trace.get("preparation_outputs", []))

    pairs = {}
    for at, producer in enumerate(graph.events[:-1]):
        consumer = graph.events[at + 1]
        if (
            producer.get("op") != "aten.bitwise_and.Tensor"
            or consumer.get("op") != "aten.ne.Scalar"
            or len(producer["outputs"]) != 1
            or len(consumer["inputs"]) != 1
            or len(consumer["outputs"]) != 1
        ):
            continue
        intermediate = producer["outputs"][0]
        consumed = consumer["inputs"][0]
        target = consumer["outputs"][0]
        if (
            key(intermediate) != key(consumed)
            or users[key(intermediate)] != [at + 1]
            or writers[key(intermediate)] != [at]
            or key(intermediate) in observed
            or intermediate["shape"] != consumed["shape"]
            or intermediate["shape"] != target["shape"]
            or rows(intermediate) != rows(target)
            or not intermediate["contiguous"]
            or not target["contiguous"]
            or target.get("dtype") != "torch.bool"
            or target["element_bytes"] != 1
        ):
            continue
        pairs[at] = target
    return pairs


def training_batch_norm_relu_candidates(graph):
    """Find exact adjacent training-BN/ReLU pairs with one logical consumer."""
    if not hasattr(graph, "trace"):
        return {}
    from .normalization_timing import current_statistics

    users = defaultdict(list)
    writers = defaultdict(list)
    for at, event in enumerate(graph.events):
        for spec in event["inputs"]:
            users[key(spec)].append(at)
        for spec in event["outputs"]:
            writers[key(spec)].append(at)
    observed = {
        key(spec)
        for checkpoint in graph.maps.get("checkpoints", [])
        for spec in ([checkpoint["spec"]] if "spec" in checkpoint else [])
        + checkpoint.get("readout_inputs", [])
    }
    observed.update(key(spec) for spec in graph.trace.get("preparation_outputs", []))

    pairs = {}
    for at, producer in enumerate(graph.events[:-1]):
        consumer = graph.events[at + 1]
        if (
            not current_statistics(producer)
            or consumer.get("op") not in ("aten.relu.default", "aten.relu_.default")
            or len(producer["outputs"]) != 1
            or len(consumer["inputs"]) != 1
            or len(consumer["outputs"]) != 1
        ):
            continue
        source = producer["outputs"][0]
        read = consumer["inputs"][0]
        target = consumer["outputs"][0]
        if (
            key(source) != key(read)
            or users[key(source)] != [at + 1]
            or writers[key(source)] != [at]
            or writers[key(target)] != [at + 1]
            or key(source) in observed
        ):
            continue
        fields = ("storage", "address", "shape", "stride", "element_bytes", "dtype", "bytes")
        if any(source[field] != read[field] for field in fields):
            continue
        # ReLU may advance an in-place version, but it must cover the same
        # dense FP32 allocation so the BN row window can write the final bits.
        if (
            source["storage"] != target["storage"]
            or any(source[field] != target[field] for field in fields[1:])
            or len(source["shape"]) != 2
            or not source["contiguous"]
            or not target["contiguous"]
            or source["element_bytes"] != 4
        ):
            continue
        pairs[at] = target
    return pairs


def training_batch_norm_add_relu_candidates(graph):
    """Find BN -> residual add -> ReLU chains that can share one row window."""
    if not hasattr(graph, "trace"):
        return {}
    from .normalization_timing import current_statistics

    users = defaultdict(list)
    writers = defaultdict(list)
    for at, event in enumerate(graph.events):
        for spec in event["inputs"]:
            users[key(spec)].append(at)
        for spec in event["outputs"]:
            writers[key(spec)].append(at)
    observed = {
        key(spec)
        for checkpoint in graph.maps.get("checkpoints", [])
        for spec in ([checkpoint["spec"]] if "spec" in checkpoint else [])
        + checkpoint.get("readout_inputs", [])
    }
    observed.update(key(spec) for spec in graph.trace.get("preparation_outputs", []))

    chains = {}
    for at, producer in enumerate(graph.events[:-2]):
        add = graph.events[at + 1]
        relu = graph.events[at + 2]
        if (
            not current_statistics(producer)
            or add.get("op") not in ("aten.add.Tensor", "aten.add_.Tensor")
            or relu.get("op") not in ("aten.relu.default", "aten.relu_.default")
            or len(producer["outputs"]) != 1
            or len(add["inputs"]) != 2
            or len(add["outputs"]) != 1
            or len(relu["inputs"]) != 1
            or len(relu["outputs"]) != 1
        ):
            continue
        source = producer["outputs"][0]
        source_positions = [i for i, spec in enumerate(add["inputs"]) if key(spec) == key(source)]
        if len(source_positions) != 1:
            continue
        residual = add["inputs"][1 - source_positions[0]]
        added = add["outputs"][0]
        read = relu["inputs"][0]
        target = relu["outputs"][0]
        if (
            users[key(source)] != [at + 1]
            or writers[key(source)] != [at]
            or key(added) != key(read)
            or users[key(added)] != [at + 2]
            or writers[key(added)] != [at + 1]
            or writers[key(target)] != [at + 2]
            or key(source) in observed
            or key(added) in observed
        ):
            continue
        fields = ("shape", "stride", "element_bytes", "dtype", "bytes")
        if (
            added["storage"] != target["storage"]
            or any(added[field] != target[field] for field in ("address",) + fields)
            or any(source[field] != added[field] for field in fields)
            or any(source[field] != residual[field] for field in fields)
            or len(source["shape"]) != 2
            or any(not spec["contiguous"] for spec in (source, residual, added, target))
            or source["element_bytes"] != 4
        ):
            continue
        chains[at] = (target, residual)
    return chains


def join(sim, index, event, last_chunk, control):
    """Retire the eliminated comparison descriptor after the fused store."""
    ready = max(sim.cycle, sim.memory.clock, sim.stats["last_tile_completion"])
    sim.cycle = ready + sim.c["issue_cycles"]
    sim.memory.clock = sim.cycle
    sim.stats["integer_mask_fusion_joins"] += 1
    sim.stages[event["stage"]] += sim.c["issue_cycles"]
    if sim.c.get("profile_events", False):
        sim.event_profile.append(
            dict(
                index=index,
                stage=event["stage"],
                kind="integer_mask_fusion_join",
                op=event["op"],
                rows=0,
                shape=event["outputs"][0]["shape"],
                cycles=sim.c["issue_cycles"],
                memory=[0] * 8,
                statistics=dict(integer_mask_fusion_joins=1),
            )
        )
    if index in sim.diagnostic_producers:
        sim.diagnostic_completions[index] = sim.cycle
    if last_chunk:
        sim.observe(index, control)
    sim.checkpoint_fork.join(sim, index, last_chunk)


def training_batch_norm_relu_join(sim, index, event, last_chunk, control):
    """Retire the eliminated ReLU descriptor after the fused final store."""
    ready = max(sim.cycle, sim.memory.clock, sim.stats["last_tile_completion"])
    sim.cycle = ready + sim.c["issue_cycles"]
    sim.memory.clock = sim.cycle
    sim.stats["training_batch_norm_relu_joins"] += 1
    sim.stages[event["stage"]] += sim.c["issue_cycles"]
    if sim.c.get("profile_events", False):
        sim.event_profile.append(
            dict(
                index=index,
                stage=event["stage"],
                kind="training_batch_norm_relu_join",
                op=event["op"],
                rows=0,
                shape=event["outputs"][0]["shape"],
                cycles=sim.c["issue_cycles"],
                memory=[0] * 8,
                statistics=dict(training_batch_norm_relu_joins=1),
            )
        )
    if index in sim.diagnostic_producers:
        sim.diagnostic_completions[index] = sim.cycle
    if last_chunk:
        sim.observe(index, control)
    sim.checkpoint_fork.join(sim, index, last_chunk)


def training_batch_norm_add_relu_join(sim, index, event, last_chunk, control):
    """Retire one eliminated residual-chain descriptor after the final store."""
    ready = max(sim.cycle, sim.memory.clock, sim.stats["last_tile_completion"])
    sim.cycle = ready + sim.c["issue_cycles"]
    sim.memory.clock = sim.cycle
    kind = event["training_batch_norm_add_relu_fusion_join"]
    counter = f"training_batch_norm_add_relu_{kind}_joins"
    sim.stats[counter] += 1
    sim.stages[event["stage"]] += sim.c["issue_cycles"]
    if sim.c.get("profile_events", False):
        sim.event_profile.append(
            dict(
                index=index,
                stage=event["stage"],
                kind=f"training_batch_norm_add_relu_{kind}_join",
                op=event["op"],
                rows=0,
                shape=event["outputs"][0]["shape"],
                cycles=sim.c["issue_cycles"],
                memory=[0] * 8,
                statistics={counter: 1},
            )
        )
    if index in sim.diagnostic_producers:
        sim.diagnostic_completions[index] = sim.cycle
    if last_chunk:
        sim.observe(index, control)
    sim.checkpoint_fork.join(sim, index, last_chunk)
