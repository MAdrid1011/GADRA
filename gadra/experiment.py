"""Repeatable nine-case experiment with fixed-resource full-factorial ablations."""
import json
import os
from pathlib import Path
import time
import subprocess
from collections import Counter

import numpy as np

from .graph import Graph,rows,key
from .engine import Simulator
from .predictor import (DemandLUT,select_checkpoints,response_record_words,
                        selector_storage_bytes)
from .baseline import project,reference_metadata

MODELS=('SceneSplat','Chorus','EmbodiedSplat')
DATASETS=('scannet','scannetpp','matterport3d')


def matrix_arithmetic_floor_ms(graph,config,live=None):
    """Optimistic unpadded array-work bound, weighted by each product dtype."""
    weighted=0
    for i,event in enumerate(graph.events):
        if not event['outputs']:continue
        precision=config['fp32_mac_cycles'] if event['outputs'][0]['element_bytes']==4 else 1
        weighted+=graph.flops(i,None if live is None else live[i])*precision
    peak=2*config['arrays']*config['array_rows']*config['array_cols']*config['clock_hz']
    return weighted/peak*1000


def anchor(graph,baseline,config):
    flops=sum(graph.flops(i) for i in range(len(graph.events)))
    raw_read=sum(s['bytes'] for e in graph.events for s in e['inputs'])
    raw_write=sum(s['bytes'] for e in graph.events for s in e['outputs'])
    return dict(model=graph.trace['model'],dataset=graph.trace['dataset'],flops=flops,
        input_rows=graph.trace['output_shape'][0],logical_tensor_read_bytes=raw_read,
        logical_tensor_write_bytes=raw_write,peak_arithmetic_floor_ms=matrix_arithmetic_floor_ms(graph,config),
        scope='Static real event shapes and actual sparse edge maps; no average per-layer cost',
        orin_reference=baseline)


def run_case(model,dataset,config_path,raw_directory,result_directory,reference=None,training_raw=None,variant_names=None,*,case_tag=None,calibration_records=None):
    requested=('none','A','B','AB') if variant_names is None else tuple(variant_names)
    if not requested or set(requested)-{'none','A','B','AB'}:
        raise ValueError('Expected one or more of none/A/B/AB')
    begin=time.perf_counter();config=json.loads(Path(config_path).read_text());raw=Path(raw_directory)
    from .profile import validate
    profile=validate(config,sensitivity=config.get('sensitivity_study',False))
    revision=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    tag=case_tag or f'{model}_{dataset}_s0'
    graph=Graph(raw/f'{tag}.json')
    if graph.trace.get('diagnostic_definition')!='l1-cumulative-v1':
        raise ValueError('Recapture with the manuscript diagnostic definition and frozen readout identity')
    obs=dict(np.load(raw/f'{tag}_observations.npz'))
    if obs['scores'].shape[-1]>config['max_queries']:
        raise ValueError('The active query set exceeds the configured history width')
    if config.get('tail_rank')!=graph.trace.get('tail_rank'):
        raise ValueError('Frozen tail rank must match the captured arithmetic and original quality target')
    if config.get('tail_rank') is not None and reference is None:
        raise ValueError('Ranked arithmetic needs an explicit software reference; historical native counters cannot project the changed graph')
    if graph.trace.get('variant') in ('common_ancestor_full','common_ancestor_stripes') and reference is None:
        raise ValueError('Ancestor arithmetic needs an explicit software reference; native counters do not describe the changed graph')
    if config.get('diagnostic_query_fp16',False)!=bool(graph.trace.get('diagnostic_query_fp16',False)):
        raise ValueError('Diagnostic precision must match the actual captured observations')
    if config.get('diagnostic_factorized_query',False)!=bool(graph.trace.get('diagnostic_factorized_query',False)):
        raise ValueError('Diagnostic factorization must match the actual captured observations')
    if config.get('embodied_native_previous',False)!=bool(graph.trace.get('embodied_native_previous',False)):
        raise ValueError('Native preceding readout must match the actual captured observations')
    if config.get('terminal_latent_query',False)!=bool(graph.trace.get('terminal_latent_query',False)):
        raise ValueError('Terminal latent query must match the actual captured observations')
    if config.get('diagnostic_transposed_composition',False)!=bool(graph.trace.get('diagnostic_transposed_composition',False)):
        raise ValueError('Query composition layout must match the captured FP32/FP16 observations')
    if config.get('diagnostic_output_fp16',False)!=bool(graph.trace.get('diagnostic_output_fp16',False)):
        raise ValueError('Early output precision must match the measured output quality')
    if config.get('diagnostic_partitioned_query',False)!=bool(graph.trace.get('diagnostic_partitioned_query',False)):
        raise ValueError('Partitioned query arithmetic must match independently captured controls')
    if config.get('diagnostic_partitioned_output',False)!=bool(graph.trace.get('diagnostic_partitioned_output',False)):
        raise ValueError('Partitioned early output must match independently captured quality')
    # Spatial row provenance belongs to the native graph, independently of
    # which semantic observations A actually executes.
    graph.maps['coverage_checkpoints']=graph.maps['checkpoints']
    from .checkpoints import select as paper_checkpoints
    obs,graph.maps['checkpoints']=paper_checkpoints(model,obs,graph.maps['checkpoints'])
    if reference:
        baseline=reference_metadata(json.loads(Path(reference).read_text()))
        if baseline.get('model')!=model or baseline.get('dataset')!=dataset:
            raise ValueError('Software reference belongs to a different workload')
        if (baseline.get('status')!='measured' or baseline.get('Orin_measured') is not True
                or baseline.get('reference_method')!='orin_cuda_events'):
            raise ValueError('Expected board-measured Orin CUDA-event timing')
        full_quality=graph.trace.get('quality',{})
    else:
        measurement=json.loads((Path(result_directory)/f'{tag}_gpu.json').read_text())
        baseline=project(graph,measurement,config)
        full_quality=measurement['quality']
    static=anchor(graph,baseline,config)
    destination=Path(result_directory)/f'{tag}.json'
    destination.with_name(tag+'_anchor.json').write_text(json.dumps(static,indent=2)+'\n')
    calibration=Path(training_raw) if training_raw else raw
    if calibration_records is None:
        raise ValueError('Paper replay requires fixed independent calibration records from a study manifest')
    from .study import calibration_observations
    training,training_contexts=calibration_observations(model,calibration_records,config)
    predictor=DemandLUT(training,config,training_contexts)
    predictor_bytes=predictor.storage_bytes()
    observation_begin=predictor.observation_begin() if config.get('gate_unreachable_checkpoints',False) else 0
    graph.maps['observation_begin']=observation_begin
    observation_end=(predictor.observation_end() if config.get('gate_unreachable_checkpoints',False)
                     else len(graph.maps['checkpoints'])-1)
    graph.maps['observation_end']=observation_end
    # Use the actual coarsest shared supply, rather than a hard-coded depth.
    # This reduces the complete-pilot dependency footprint and increases the
    # number of consumers served by one completed context response.
    context_checkpoint=min(graph.maps['checkpoints'],key=lambda cp:rows(cp['spec']))
    contexts=context_checkpoint['mapping']
    graph.maps['response_context']=context_checkpoint
    stops,witnesses,quality=predictor.decide(obs,contexts)
    predictor_edges=predictor.edges
    response_edges=predictor.response_edges
    nearest_edges=getattr(predictor,'nearest_edges',None)
    tolerance=min(config['semantic_tolerance'],config.get('class_repair_tolerance',config['semantic_tolerance']))
    response_actions=[table<=tolerance for table in predictor.response_table]
    nearest_actions=[table<=tolerance for table in getattr(predictor,'nearest_table',[])]
    gate_support=predictor.feedback_gate_support() if config.get('diagnostic_feedback_gate',False) else None
    quality['passes_thresholds']=(quality['class_agreement']>=config['quality_min_class_agreement'] and
                                  quality['within_tolerance']>=config['quality_min_within_tolerance'])
    del training,predictor
    # Keep failing experiments visible; evaluation labels never change a stop.
    full=np.full(len(stops),len(graph.maps['checkpoints'])-1,dtype=np.int16)
    root=graph.events[-1]['outputs'][0]
    full_live=graph.closure([(root,np.arange(rows(root)))])
    first=graph.closure(graph.checkpoint_seeds(stops,witnesses))
    total=graph.closure(graph.checkpoint_seeds(stops))
    second=total.minus(first)
    kept=sum(graph.flops(i,x) for i,x in enumerate(total))
    reference_flops=sum(graph.flops(i,x) for i,x in enumerate(full_live))
    conservative_dependencies=dict(graph.conservative)
    geometry=raw if (raw/f'{tag}_coverage.npz').exists() else calibration
    cover=np.load(geometry/f'{tag}_coverage.npz');order=cover['order']
    cover_meta=json.loads((geometry/f'{tag}_coverage.json').read_text())
    prepared=None;preparation_floor_ms=0
    preparation=graph.trace.get('query_preparation_trace')
    if preparation:
        from .screen import replay_trace
        prepared,_=replay_trace(raw/preparation,config)
        prep_graph=Graph(raw/preparation)
        prep_outputs=prep_graph.trace.get('preparation_outputs',[prep_graph.events[-1]['outputs'][0]])
        prep_live=prep_graph.closure([(s,np.arange(rows(s))) for s in prep_outputs])
        preparation_floor_ms=matrix_arithmetic_floor_ms(prep_graph,config,prep_live)
        static['preparation']=dict(trace=preparation,flops=prepared['reference_flops'],
            peak_arithmetic_floor_ms=preparation_floor_ms,cycles=prepared['cycles'])
        static['scope']+='; encoding anchor above, common cold preparation listed separately'
        destination.with_name(tag+'_anchor.json').write_text(json.dumps(static,indent=2)+'\n')
    variants={}
    for a,b in ((False,False),(True,False),(False,True),(True,True)):
        name=('A' if a else '')+('B' if b else '') or 'none'
        if name not in requested:continue
        deployed=config
        response_entries=64 if a else 0
        selector_bytes=selector_storage_bytes(rows(context_checkpoint['spec']),len(stops),config) if response_entries else 0
        from .diagnostic_pipeline import feedback_gate_storage,feedback_gate_header_bytes
        gate_bytes=feedback_gate_storage(rows(context_checkpoint['spec']),len(stops),config) if a else 0
        joint_bytes=(rows(context_checkpoint['spec'])*config.get('witnesses_per_context',1)*4
                     if a and config.get('predictor_joint_response',False) else 0)
        if a:
            # The frozen tables share the declared control SRAM with B's
            # finite ready-window metadata, rather than adding free storage.
            deployed=dict(config,control_sram_bytes=config['control_metadata_bytes']-predictor_bytes-
                          64*32-selector_bytes-gate_bytes-joint_bytes)
            if deployed['control_sram_bytes']<=0:raise ValueError('No control SRAM remains after predictor tables')
        links={}
        if b and config.get('coverage_linked_supply',False):
            if not {'link_offsets','link_neighbors'}<=set(cover.files):
                raise ValueError('Linked supply requires captured ellipsoid links; regenerate coverage archive')
            links['coverage_links']=(cover['link_offsets'],cover['link_neighbors'])
        sim=Simulator(graph,deployed,coverage=b,order=order,demand=a,**links)
        sim.predictor_edges=predictor_edges
        sim.predictor_response_edges=response_edges
        sim.predictor_gate_support=gate_support
        sim.predictor_nearest_edges=nearest_edges
        sim.predictor_response_actions=response_actions
        sim.predictor_nearest_actions=nearest_actions
        if joint_bytes:sim.stats['joint_owner_mapping_reserved_bytes']=joint_bytes
        sim.stats['witness_selector_reserved_bytes']=selector_bytes
        if a:
            sim.install_predictor(predictor_bytes+(feedback_gate_header_bytes(config) if gate_bytes else 0))
            if response_entries:sim.initialize_responses(response_entries)
            if config.get('retain_phase_suppliers',False):
                # Full-depth fallback is the superset of every possible
                # causal exit. Protect its incremental input cut before any
                # pilot feedback is visible; never use target-row stops here.
                maximum_total=graph.closure(graph.checkpoint_seeds(full))
                remaining_full=maximum_total.minus(first)
                sim.phase_suppliers={key(spec) for i,ids in enumerate(remaining_full) if len(ids)
                    for spec,_ in graph.inputs_for(i,ids) if spec['role']=='feature'}
                sim.phase_suppliers.update(old for old,_,tokens in remaining_full.carry.values() if len(tokens))
                sim.phase_suppliers.update(key(spec) for cp in graph.maps['checkpoints'][observation_begin:observation_end]
                    for spec in cp['readout_inputs'])
                if config.get('retire_private_phase_temporaries',False):
                    carry={identity for old,new,_ in remaining_full.carry.values() for identity in (old,new)}
                    carry.update(identity for old,new,_ in first.carry.values() for identity in (old,new))
                    private=sim.private_phase_temporaries(carry)
                    sim.stats['phase_private_unprotected_versions']=len(sim.phase_suppliers & private)
                    sim.phase_suppliers.difference_update(private)
                sim.stats['phase_supplier_protected_versions']=len(sim.phase_suppliers)
            if b:sim.install_coverage_preparation(cover_meta)
            sim.execute(first,control=(obs,stops,witnesses,'pilot'))
            if config.get('finite_diagnostic_pipeline',False):
                from .diagnostic_pipeline import complete_feedback
                complete_feedback(sim,obs,witnesses)
                # Compare the streaming construction against the offline
                # finished-pilot label, never unfinished-position scores.
                from .predictor import remaining_correction
                pilot={k:v[:,witnesses] for k,v in obs.items()
                    if v.ndim>=2 and v.shape[:2]==obs['scores'].shape[:2]}
                expected=remaining_correction(pilot,config.get('correction_horizon','suffix'))[sim.completed_response_depths]
                np.testing.assert_allclose(sim.completed_response_values,expected,rtol=1e-6,atol=1e-7)
            # All pilot results are complete before any dependent stop uses
            # feedback. Cache state and spill cost carry into phase 2.
            pilot_completed=int(sim.cycle)
            if config.get('retain_phase_suppliers',False):sim.phase_suppliers=set()
            sim.execute(second,control=(obs,stops,witnesses,'remaining'))
            if hasattr(sim,'joint_score_payload'):
                from .diagnostic_pipeline import retire
                retire(sim,sim.joint_score_payload)
            expected_exits=int((stops<len(graph.maps['checkpoints'])-1).sum())
            expected_observations=sum(int(((stops>=depth)&~witnesses).sum())
                                      for depth in range(observation_begin,observation_end))
            expected_observations+=int(witnesses.sum())*(len(graph.maps['checkpoints'])-1-observation_begin)
            if sim.stats['early_output_rows']!=expected_exits:
                raise AssertionError(('Incomplete early output contract',sim.stats['early_output_rows'],expected_exits))
            if (sim.stats['diagnostic_leaf_observations']+sim.stats['feedback_gate_skipped_observations']+
                    sim.stats['feedback_skipped_pilot_observations']!=expected_observations):
                raise AssertionError(('Missing control observations',sim.stats['diagnostic_leaf_observations'],expected_observations))
        else:
            if b:sim.install_coverage_preparation(cover_meta)
            sim.execute(full_live);pilot_completed=None
        result=sim.finish(cover_meta if b else None)
        result['quality_pass']=quality['passes_thresholds'] if a else True
        peak=2*config['arrays']*config['array_rows']*config['array_cols']*config['clock_hz']
        arithmetic_cycles=matrix_arithmetic_floor_ms(graph,config,total if a else full_live)*config['clock_hz']/1000
        diagnostic_flops=result['statistics'].get('diagnostic_flops',0)
        half_flops=(result['statistics'].get('diagnostic_query_fp16_flops',0)+
                    result['statistics'].get('diagnostic_output_fp16_flops',0))
        arithmetic_cycles+=((diagnostic_flops-half_flops)*config['fp32_mac_cycles']+half_flops)/peak*config['clock_hz']
        result['optimistic_arithmetic_floor_ms']=arithmetic_cycles/config['clock_hz']*1000
        result['optimistic_arithmetic_speedup_ceiling']=baseline['orin_reference_ms']/result['optimistic_arithmetic_floor_ms']
        result['speedup_vs_orin_reference']=baseline['orin_reference_ms']/result['ms']
        result['pilot_completed_cycle']=pilot_completed
        if prepared is not None:
            # Reuse the cold, fully drained preparation composition used by
            # captured-program timing; tables and prepared weights are never
            # inherited into the cold encoding hierarchy for free.
            from .screen import include_preparation
            result['optimistic_arithmetic_floor_ms']+=preparation_floor_ms
            include_preparation(result,prepared,config)
            result['optimistic_arithmetic_speedup_ceiling']=baseline['orin_reference_ms']/result['optimistic_arithmetic_floor_ms']
            result['speedup_vs_orin_reference']=baseline['orin_reference_ms']/result['ms']
        variants[name]=result
        print(tag,list(variants)[-1],round(result['ms'],3),'ms',flush=True)
    report=dict(model=model,dataset=dataset,profile=profile,static_anchor=static,baseline=baseline,variants=variants,
        quality=quality,full_depth_quality=full_quality,
        variant=graph.trace.get('variant'),reference_source=str(reference) if reference else None,
        calibration_source=str(calibration),coverage_source=str(geometry),
        pruning=dict(reference_flops=reference_flops,kept_flops=kept,
                     removed_fraction=1-kept/reference_flops,conservative_dependencies=conservative_dependencies),
        predictor_split='Fixed independent 30-scene model calibration, pooled across three datasets',
        response_context_checkpoint=context_checkpoint['name'],
        observation_begin=observation_begin,
        observation_end=observation_end,
        predictor_storage_bytes=predictor_bytes,
        dependency_granule_storage_counts=dict(Counter(graph.dependency_granules.values())),
        simulator_seconds=time.perf_counter()-begin,config=config,
        implementation_revision=revision,
        working_tree_modified=subprocess.run(['git','diff','--quiet']).returncode!=0,
        numba_cache_namespace=os.environ.get('NUMBA_CACHE_DIR'),
        notes=['One explicitly identified fragment in the complete scene manifest',
               'A quality results are measured from exact retained checkpoint values with supply closure',
               'No PSNR/SSIM/LPIPS: output is semantic vectors, no renderer in this scope'])
    destination.write_text(json.dumps(report,indent=2)+'\n')
    return str(destination)
