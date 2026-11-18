"""Thin adapters to the already verified real-model runtimes."""
from contextlib import nullcontext
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = Path(
    os.environ.get("GADRA_WORKLOADS_ROOT", ROOT / "local" / "workloads")
).expanduser().resolve()
sys.path.insert(0, str(WORKLOADS / 'scripts'))


INPUT_SCOPES = ('chunk-first', 'scene-first', 'scene-all')


def select_fragment(scene, data, settings, input_scope, fragment_index=0):
    """Select real native rows without changing the established default input."""
    if input_scope not in INPUT_SCOPES:
        raise ValueError(f'Unknown input scope: {input_scope}')
    selected = (next(scene.spatial_chunks(data, settings['max_points'],
                    settings['chunks_per_scene'], settings['seed']))
                if input_scope == 'chunk-first' else data)
    fragments=iter(scene.fragments(selected))
    for _ in range(fragment_index+1):
        try:indices=next(fragments)
        except StopIteration:raise ValueError('Fragment index exceeds the native scene partition') from None
    metadata = dict(input_scope=input_scope, fragment_index=fragment_index,
                    encoded_rows=len(indices), raw_gaussians=selected['raw_count'],
                    representatives_1cm=selected['representatives'],
                    fragment_count=int(selected['counts'].max()),
                    full_scene_raw_gaussians=data['raw_count'])
    if input_scope == 'chunk-first':
        metadata.update(chunk_id=selected['chunk_id'],
                        available_chunks=selected['available_chunks'])
    return selected, indices, metadata


def load_case(family, dataset, index, *, input_scope='chunk-first',role='evaluation',fragment_index=0,scene_id=None):
    if input_scope not in INPUT_SCOPES:
        raise ValueError(f'Unknown input scope: {input_scope}')
    if role not in ('evaluation','calibration','tuning'):
        raise ValueError(f'Unknown workload role: {role}')
    if family == 'EmbodiedSplat':
        if fragment_index:
            raise ValueError('Each EmbodiedSplat native-cache entry is a complete inference unit')
        sys.path.insert(0, str(WORKLOADS / 'build/minkowski_graph_backend'))
        from measure_embodiedsplat_native import entries, prepare, model_for
        entry = entries(role, dataset)[index]
        record, coordinates, features, inverse, _ = prepare(WORKLOADS / entry['cache'])
        model = model_for(record)
        # prepare() constructs exactly one collated scene. Rechecking this via
        # decomposed_features materializes unused indexed feature copies.
        model.memory._single_scene_prevalidated = True
        model._gadra_native = (record, coordinates, features, inverse)
        model._gadra_input_scope=dict(input_scope='native-cache',scene_id=scene_id or entry.get('scene_id',str(index)),fragment_index=0,fragment_count=1,encoded_rows=len(inverse))
        return model, lambda: model(coordinates, features, inverse=inverse), nullcontext(), entry
    import scenesplat_runtime as scene
    import chorus_runtime as chorus
    from chorus_serialization import fused_serialization
    assets = json.loads((WORKLOADS / 'results/assets.json').read_text())
    settings = json.loads((WORKLOADS / 'results/scenesplat_fp16.json').read_text())['settings']
    entry = [r for r in assets['datasets'][dataset]['scenes'] if r['role'] == role][index]
    data = scene.prepare_scene(WORKLOADS / entry['local'], settings['seed'])
    chunk, indices, scope = select_fragment(scene, data, settings, input_scope,fragment_index)
    scope['scene_id']=scene_id or entry.get('scene_id',entry['local'])
    inputs = scene.gpu_input(chunk, indices, half=True)
    if family == 'SceneSplat':
        model = scene.optimize_inference(scene.load_model(WORKLOADS / 'checkpoints' / assets['model']['file']), half=True)
        runtime, context = scene, nullcontext()
    else:
        checkpoint = json.loads((WORKLOADS / 'results/chorus_assets.json').read_text())['model']['local']
        model = chorus.load_model(checkpoint, half=True)
        runtime, context = chorus, fused_serialization()
    model._gadra_inputs = inputs
    model._gadra_chunk = chunk
    model._gadra_fragment_indices = indices
    model._gadra_input_scope = scope
    return model, lambda: runtime.encode(model, inputs, settings['seed']), context, entry


def gpu_baseline(family, dataset, index, repeats, *, input_scope='scene-all',role='evaluation',fragment_index=0,scene_id=None):
    import fcntl
    import time
    import uuid
    import subprocess
    import torch
    import torch.nn.functional as F
    lock = (WORKLOADS / 'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(4)
    if repeats<1:raise ValueError('Positive measurement repeats are required')
    if 'orin' not in torch.cuda.get_device_name().lower():
        raise ValueError('Capture the GPU comparison on the AGX Orin board')
    model, call, context, entry = load_case(family, dataset, index, input_scope=input_scope,role=role,fragment_index=fragment_index,scene_id=scene_id)
    input_metadata = getattr(model, '_gadra_input_scope', dict(input_scope='native-cache'))
    with torch.inference_mode(), context:
        for _ in range(2):
            reference = call()
        samples = []
        for _ in range(repeats):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start.record()
            output = call()
            end.record()
            end.synchronize()
            samples.append(dict(wall_ms=(time.perf_counter() - begin) * 1000, gpu_ms=start.elapsed_time(end)))
        diff = (output.float() - reference.float()).abs()
        quality = dict(max_abs_error=float(diff.max()), rmse=float(diff.square().mean().sqrt()),
                       mean_cosine=float(F.cosine_similarity(output.float(), reference.float()).mean()),
                       bitwise_equal=torch.equal(reference, output))
    del model
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return dict(model=family, dataset=dataset, scene_index=index, entry=entry,
                output_shape=list(output.shape), device=torch.cuda.get_device_name(),
                samples=samples, quality=quality, measured=True,
                run_id=str(uuid.uuid4()),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                platform=(Path('/proc/device-tree/model').read_bytes().decode().strip('\x00\n')
                          if Path('/proc/device-tree/model').exists() else torch.cuda.get_device_name()),
                input_metadata=input_metadata,
                scope='Complete 3D semantic encoding; input preparation and checkpoint I/O excluded')
