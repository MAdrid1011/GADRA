"""Causal checkpoint replay with immutable admission and decision records."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

from .graph import Graph,rows
from .engine import Simulator
from .predictor import DemandLUT,witness_layout,selector_storage_bytes
from .baseline import project,reference_metadata

MODELS=('SceneSplat','Chorus','EmbodiedSplat')
DATASETS=('scannet','scannetpp','matterport3d')


def matrix_arithmetic_floor_ms(graph,config,live=None):
    weighted=0
    for i,event in enumerate(graph.events):
        if not event['outputs']:continue
        precision=config['fp32_mac_cycles'] if event['outputs'][0]['element_bytes']==4 else 1
        weighted+=graph.flops(i,None if live is None else live[i])*precision
    peak=2*config['arrays']*config['array_rows']*config['array_cols']*config['clock_hz']
    return weighted/peak*1000


def anchor(graph,baseline,config):
    return dict(model=graph.trace['model'],dataset=graph.trace['dataset'],
        flops=sum(graph.flops(i) for i in range(len(graph.events))),
        input_rows=graph.trace['output_shape'][0],
        logical_tensor_read_bytes=sum(s['bytes'] for e in graph.events for s in e['inputs']),
        logical_tensor_write_bytes=sum(s['bytes'] for e in graph.events for s in e['outputs']),
        peak_arithmetic_floor_ms=matrix_arithmetic_floor_ms(graph,config),
        orin_reference=baseline)


def run_case(model,dataset,config_path,raw_directory,result_directory,reference=None,
             training_raw=None,variant_names=None,*,case_tag=None,calibration_records=None):
    from .profile import validate
    from .checkpoints import select
    from .study import calibration_observations,digest
    from .online import execute_online
    from .quality import at_committed_depths
    requested=('none','A','B','AB') if variant_names is None else tuple(variant_names)
    if not requested or set(requested)-{'none','A','B','AB'}:
        raise ValueError('Expected one or more of none/A/B/AB')
    begin=time.perf_counter()
    config=json.loads(Path(config_path).read_text())
    profile=validate(config,sensitivity=config.get('sensitivity_study',False))
    raw=Path(raw_directory)
    output=Path(result_directory)
    output.mkdir(parents=True,exist_ok=True)
    tag=case_tag or f'{model}_{dataset}_s0'
    graph=Graph(raw/f'{tag}.json')
    if (graph.trace.get('model'),graph.trace.get('dataset'))!=(model,dataset):
        raise ValueError('The captured workload identity differs from the requested case')
    if graph.trace.get('diagnostic_definition')!='l1-cumulative-v1':
        raise ValueError('Capture the frozen manuscript diagnostic definition before replay')
    for flag in ('diagnostic_query_fp16','diagnostic_factorized_query',
                 'embodied_native_previous','terminal_latent_query',
                 'diagnostic_transposed_composition','diagnostic_output_fp16',
                 'diagnostic_partitioned_query','diagnostic_partitioned_output'):
        if graph.trace.get(flag,False):
            raise ValueError(f'The canonical FP32 diagnostic path does not enable {flag}')
    if graph.trace.get('variant') or graph.trace.get('tail_rank') is not None:
        raise ValueError('Canonical replay uses the unmodified native model graph')
    observations=dict(np.load(raw/f'{tag}_observations.npz'))
    if observations['scores'].shape[-1]>config['max_queries']:
        raise ValueError('The query set exceeds the declared history width')
    graph.maps['coverage_checkpoints']=graph.maps['checkpoints']
    observations,graph.maps['checkpoints']=select(model,observations,graph.maps['checkpoints'])
    graph.maps['observation_begin']=0
    graph.maps['observation_end']=len(graph.maps['checkpoints'])-1
    if reference:
        baseline=reference_metadata(json.loads(Path(reference).read_text()))
        if (baseline.get('model'),baseline.get('dataset'))!=(model,dataset):
            raise ValueError('The board measurement belongs to another workload')
        for field in ('scene_id','fragment_index','encoded_rows'):
            expected=graph.trace.get('input_metadata',{}).get(field)
            if expected is not None and baseline.get('input_metadata',{}).get(field)!=expected:
                raise ValueError(f'The board measurement disagrees on {field}')
    else:
        measurement=json.loads((output/f'{tag}_gpu.json').read_text())
        baseline=project(graph,measurement,config)
    if calibration_records is None:
        raise ValueError('Fixed independent calibration records are required')
    training,training_contexts=calibration_observations(model,calibration_records,config)
    predictor=DemandLUT(training,config,training_contexts)
    predictor_bytes=predictor.storage_bytes()
    context=min(graph.maps['checkpoints'],key=lambda cp:rows(cp['spec']))
    graph.maps['response_context']=context
    _,owners=witness_layout(np.asarray(context['mapping']),config)
    count=len(context['mapping'])
    witnesses=np.zeros(count,bool)
    witnesses[owners[owners>=0]]=True
    final_depth=len(graph.maps['checkpoints'])-1
    full_stops=np.full(count,final_depth,np.int16)
    # Fixed input-derived pilot requests do not depend on predicted exits.
    pilot_plan=graph.closure(graph.checkpoint_seeds(full_stops,witnesses))
    root=graph.events[-1]['outputs'][0]
    full_live=graph.closure([(root,np.arange(rows(root)))])
    reference_flops=sum(graph.flops(i,ids) for i,ids in enumerate(full_live))
    cover=np.load(raw/f'{tag}_coverage.npz')
    coverage=json.loads((raw/f'{tag}_coverage.json').read_text())
    static=anchor(graph,baseline,config)
    (output/f'{tag}_anchor.json').write_text(json.dumps(static,indent=2)+'\n')
    variants={}
    for name in requested:
        a=name in ('A','AB')
        b=name in ('B','AB')
        deployed=config
        if a:
            selector_bytes=selector_storage_bytes(rows(context['spec']),count,config)
            available=config['control_metadata_bytes']-predictor_bytes-64*32-selector_bytes
            if available<=0:
                raise ValueError('Predictor and active metadata exceed the declared control partition')
            deployed=dict(config,control_sram_bytes=available)
        sim=Simulator(graph,deployed,coverage=b,order=cover['order'],demand=a)
        if b:sim.install_coverage_preparation(coverage)
        if a:
            sim.predictor_edges=predictor.edges
            sim.predictor_response_edges=predictor.response_edges
            sim.predictor_gate_support=None
            sim.predictor_nearest_edges=None
            sim.predictor_response_actions=[table<=config['semantic_tolerance']
                                            for table in predictor.response_table]
            sim.predictor_nearest_actions=[]
            sim.stats['witness_selector_reserved_bytes']=selector_bytes
            sim.install_predictor(predictor_bytes)
            sim.initialize_responses(64)
            state,kept=execute_online(sim,predictor,observations,witnesses,pilot_plan)
            stops=state.stops
            low_support=int(state.unsupported_positions.sum())
        else:
            sim.execute(full_live)
            sim.audit_events.append(dict(kind='native_outputs_commit',cycle=int(sim.cycle),
                positions=np.arange(count).tolist(),depth=final_depth))
            stops=full_stops.copy()
            kept=sum(graph.flops(e['event'],np.asarray(e['positions'],np.int64))
                     for e in sim.audit_events if e['kind']=='native_issue')
            low_support=0
        result=sim.finish(coverage if b else None)
        quality=at_committed_depths(observations,stops,config)
        decision_file=output/f'{tag}_{name}_decisions.npz'
        np.savez_compressed(decision_file,stop_depths=stops)
        result.update(quality=quality,main_flops=kept,reference_flops=reference_flops,
            quality_pass=quality['passes_thresholds'],
            decisions_file=decision_file.name,decisions_sha256=digest(decision_file),
            unsupported_conditional_local_candidates=low_support,
            pruning_fraction=1-kept/reference_flops if reference_flops else 0,
            decision_source=('checkpoint_committed_online_state' if a else 'native_full_depth_output'))
        variants[name]=result
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report=dict(model=model,dataset=dataset,profile=profile,config=config,
        baseline=baseline,static_anchor=static,variants=variants,
        input_trace=str(raw/f'{tag}.json'),predictor_storage_bytes=predictor_bytes,
        predictor_split='Fixed independent model calibration across three datasets',
        implementation_revision=revision,
        working_tree_modified=subprocess.run(['git','diff','--quiet']).returncode!=0,
        simulator_seconds=time.perf_counter()-begin,
        implementation_validation='requires_paired_implementation_record',
        decision_policy='Matched response selects its conditional cell; unsupported selected cells continue',
        scope='One manifest fragment. Full-scene quality is aggregated by study.py',
        numba_cache_namespace=os.environ.get('NUMBA_CACHE_DIR'))
    destination=output/f'{tag}.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    return str(destination)
