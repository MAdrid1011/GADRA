"""Repeatable nine-case experiment with fixed-resource full-factorial ablations."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import csv
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
from .baseline import project

MODELS=('SceneSplat','Chorus','EmbodiedSplat')
DATASETS=('scannet','scannetpp','matterport3d')


def numerical_report(directory):
    """Keep functional-baseline evidence separate from timing experiments."""
    directory=Path(directory);records=[]
    for model in MODELS:
        for dataset in DATASETS:
            r=json.loads((directory/f'{model}_{dataset}_s0_numerical.json').read_text())
            native=r['native_replay'];lower=r['lowered']
            records.append(dict(model=model,dataset=dataset,
                native_max_abs_error=native['max_abs_error'],
                native_class_agreement=native['class_agreement'],
                lowered_class_agreement=lower['class_agreement'],
                lowered_within_tolerance=lower['within_tolerance'],
                lowered_mean_cosine=lower['mean_cosine'],
                lowered_rmse=lower['rmse'],passes_thresholds=lower['passes_thresholds']))
    with (directory/'numerical_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    lines=['# 无机制基线数值重执行','',
        '每组合首个真实片段的完整三维语义编码。原生重执行与显式稀疏乘加均从初始输入、权重和整数几何启动；不读取捕获的浮点中间值作为输入根。',
        '', '| 模型 | 数据集 | 原生最大绝对误差 | 显式展开类别一致率 | regret≤0.01 | 平均余弦 |',
        '|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['native_max_abs_error']:.3g} | {r['lowered_class_agreement']:.4%} | {r['lowered_within_tolerance']:.4%} | {r['lowered_mean_cosine']:.8f} |")
    lines+=['','九组均满足类别一致率≥99%、regret≤0.01 比例≥99%的基线门槛。',
        '原生重执行使用原有内核；显式展开按真实 offset 边 gather/GEMM/scatter，以 FP32 产品与累加后转回输出精度。矩阵单元 RTL 浮点舍入、A/AB 选择后的完整重执行尚需后续验证。',
        '', '本轮修复：spconv Native 隐式中心和镜像边；ME kernel_map 保留权重偏移编号与空偏移；Chorus Triton 编码、ME C++ 坐标插入及 batch 分解显式记录几何边界。几何索引仍属于计费操作。',
        '此前 results/final 是旧 trace 的历史探索，不能与本轮结果混用。新数值证据不代表已经获得性能加速。']
    (directory/'NUMERICAL.md').write_text('\n'.join(lines)+'\n')
    return records


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
        orin_projection=baseline)


def run_case(model,dataset,config_path,raw_directory,result_directory,reference=None,training_raw=None,variant_names=None):
    requested=('none','A','B','AB') if variant_names is None else tuple(variant_names)
    if not requested or set(requested)-{'none','A','B','AB'}:
        raise ValueError('Expected one or more of none/A/B/AB')
    begin=time.perf_counter();config=json.loads(Path(config_path).read_text());raw=Path(raw_directory)
    revision=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    tag=f'{model}_{dataset}_s0'
    graph=Graph(raw/f'{tag}.json')
    obs=dict(np.load(raw/f'{tag}_observations.npz'))
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
    obs,graph.maps['checkpoints']=select_checkpoints(obs,graph.maps['checkpoints'],config['checkpoint_policy'])
    if reference:
        baseline=json.loads(Path(reference).read_text())
        if baseline.get('model')!=model or baseline.get('dataset')!=dataset:
            raise ValueError('Software reference belongs to a different workload')
        if not baseline.get('quality',{}).get('passes_thresholds',False):
            raise ValueError('Software reference failed quality')
        if baseline.get('status')!='projection_only' or baseline.get('Orin_measured') is not False:
            raise ValueError('Expected an explicit RTX-to-Orin software projection')
        full_quality=graph.trace.get('quality',{})
        if not full_quality.get('passes_thresholds',False):raise ValueError('Candidate trace failed quality')
    else:
        measurement=json.loads((Path(result_directory)/f'{tag}_gpu.json').read_text())
        baseline=project(graph,measurement,config)
        full_quality=measurement['quality']
    static=anchor(graph,baseline,config)
    destination=Path(result_directory)/f'{tag}.json'
    destination.with_name(tag+'_anchor.json').write_text(json.dumps(static,indent=2)+'\n')
    training=[];training_contexts=[];calibration=Path(training_raw) if training_raw else raw
    for other in DATASETS:
        if other==dataset:continue
        data=dict(np.load(calibration/f'{model}_{other}_s0_observations.npz'))
        other_graph=Graph(calibration/f'{model}_{other}_s0.json')
        if other_graph.trace.get('tail_rank')!=graph.trace.get('tail_rank'):
            raise ValueError('Tail rank differs between calibration and target')
        if bool(other_graph.trace.get('diagnostic_transposed_composition',False))!=bool(graph.trace.get('diagnostic_transposed_composition',False)):
            raise ValueError('Query composition layout differs between calibration and target')
        if ('control_scores' in data)!=('control_scores' in obs):
            raise ValueError('Control precision differs between calibration and target observations')
        data,other_checkpoints=select_checkpoints(data,other_graph.maps['checkpoints'],config['checkpoint_policy'])
        training.append(data)
        training_contexts.append(min(other_checkpoints,key=lambda cp:rows(cp['spec']))['mapping'])
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
        response_entries=rows(context_checkpoint['spec'])*len(graph.maps['checkpoints'])*config.get('witnesses_per_context',1) if a and config.get('finite_diagnostic_pipeline',False) else 0
        selector_bytes=selector_storage_bytes(rows(context_checkpoint['spec']),len(stops),config) if response_entries else 0
        from .diagnostic_pipeline import feedback_gate_storage,feedback_gate_header_bytes
        gate_bytes=feedback_gate_storage(rows(context_checkpoint['spec']),len(stops),config) if a else 0
        joint_bytes=(rows(context_checkpoint['spec'])*config.get('witnesses_per_context',1)*4
                     if a and config.get('predictor_joint_response',False) else 0)
        if a:
            # The frozen tables share the declared control SRAM with B's
            # finite ready-window metadata, rather than adding free storage.
            deployed=dict(config,control_sram_bytes=config['control_sram_bytes']-predictor_bytes-
                          response_entries*4*response_record_words(config)-selector_bytes-gate_bytes-joint_bytes)
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
                np.testing.assert_allclose(sim.completed_response_values,expected,rtol=0,atol=0)
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
        result['speedup_vs_orin_projection']=baseline['orin_reference_ms']/result['ms']
        result['speedup_vs_ideal_orin_roof']=baseline['ideal_orin_roof_ms']/result['ms']
        result['pilot_completed_cycle']=pilot_completed
        if prepared is not None:
            # Reuse the cold, fully drained preparation composition used by
            # captured-program timing; tables and prepared weights are never
            # inherited into the cold encoding hierarchy for free.
            from .screen import include_preparation
            result['optimistic_arithmetic_floor_ms']+=preparation_floor_ms
            include_preparation(result,prepared,config)
            result['optimistic_arithmetic_speedup_ceiling']=baseline['orin_reference_ms']/result['optimistic_arithmetic_floor_ms']
            result['speedup_vs_orin_projection']=baseline['orin_reference_ms']/result['ms']
            result['speedup_vs_ideal_orin_roof']=baseline['ideal_orin_roof_ms']/result['ms']
        variants[name]=result
        print(tag,list(variants)[-1],round(result['ms'],3),'ms',flush=True)
    report=dict(model=model,dataset=dataset,static_anchor=static,baseline=baseline,variants=variants,
        quality=quality,full_depth_quality=full_quality,
        variant=graph.trace.get('variant'),reference_source=str(reference) if reference else None,
        calibration_source=str(calibration),coverage_source=str(geometry),
        pruning=dict(reference_flops=reference_flops,kept_flops=kept,
                     removed_fraction=1-kept/reference_flops,conservative_dependencies=conservative_dependencies),
        predictor_split='Leave-one-dataset-out, both other datasets; no target-dataset labels in LUT',
        response_context_checkpoint=context_checkpoint['name'],
        observation_begin=observation_begin,
        observation_end=observation_end,
        predictor_storage_bytes=predictor_bytes,
        dependency_granule_storage_counts=dict(Counter(graph.dependency_granules.values())),
        simulator_seconds=time.perf_counter()-begin,config=config,
        implementation_revision=revision,
        working_tree_modified=subprocess.run(['git','diff','--quiet']).returncode!=0,
        numba_cache_namespace=os.environ.get('NUMBA_CACHE_DIR'),
        notes=['One full first fragment per combination, not full dataset evaluation',
               'A quality results are measured from exact retained checkpoint values with supply closure',
               'No PSNR/SSIM/LPIPS: output is semantic vectors, no renderer in this scope'])
    destination.write_text(json.dumps(report,indent=2)+'\n')
    return str(destination)


def run_case_parallel(model,dataset,config_path,raw_directory,result_directory,reference=None,training_raw=None,workers=4):
    """Independent ablations in separate host processes, same simulated RTL.

    Workers never share a mutable Graph, directory, predictor or SRAM state.
    Only complete consistent records are merged into the usual case output.
    """
    import shutil
    if workers<1:raise ValueError('Expected at least one host worker')
    begin=time.perf_counter();out=Path(result_directory);out.mkdir(parents=True,exist_ok=True)
    tag=f'{model}_{dataset}_s0';names=('none','A','B','AB')
    # Calibration may already have initialized HiGHS worker state in this
    # parent. Forking it can spin indefinitely inside the child MILP call.
    # Fresh interpreters also isolate all mutable simulator/JIT state.
    with ProcessPoolExecutor(max_workers=min(workers,len(names)),mp_context=get_context('spawn')) as pool:
        jobs=[]
        for name in names:
            part=out/name;part.mkdir(exist_ok=True)
            if reference is None:shutil.copyfile(out/f'{tag}_gpu.json',part/f'{tag}_gpu.json')
            jobs.append(pool.submit(run_case,model,dataset,config_path,raw_directory,str(part),reference,training_raw,(name,)))
        records=[json.loads(Path(job.result()).read_text()) for job in jobs]
    combined=records[0].copy();combined['variants']={}
    for name,record in zip(names,records):
        for field in ('model','dataset','config','baseline','quality','pruning','implementation_revision','working_tree_modified'):
            if record[field]!=combined[field]:raise AssertionError(('Inconsistent ablation',name,field))
        combined['variants'].update(record['variants'])
    combined['simulator_seconds']=time.perf_counter()-begin
    combined['host_workers']=min(workers,len(names))
    destination=out/f'{tag}.json';destination.write_text(json.dumps(combined,indent=2)+'\n')
    shutil.copyfile(out/names[0]/f'{tag}_anchor.json',out/f'{tag}_anchor.json')
    return str(destination)


def run_matrix(config_path,raw_directory,result_directory,workers):
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        jobs=[pool.submit(run_case,m,d,config_path,raw_directory,result_directory) for m in MODELS for d in DATASETS]
        for job in jobs:print('completed',job.result(),flush=True)
    report(result_directory)


def report(directory):
    directory=Path(directory);records=[];cases=[]
    for model in MODELS:
        for dataset in DATASETS:
            p=directory/f'{model}_{dataset}_s0.json'
            if not p.exists():raise ValueError(f'Incomplete nine-case matrix: {p} is missing')
            r=json.loads(p.read_text());v=r['variants'];q=r['quality']
            if set(v)!={'none','A','B','AB'}:
                raise ValueError(f'Incomplete ablations: {p}')
            cases.append(r)
            records.append(dict(model=model,dataset=dataset,rtx_ms=r['baseline']['rtx_measured_ms'],
                orin_ms=r['baseline']['orin_reference_ms'],none_cycles=v['none']['cycles'],
                A_cycles=v['A']['cycles'],B_cycles=v['B']['cycles'],AB_cycles=v['AB']['cycles'],
                none_ms=v['none']['ms'],A_ms=v['A']['ms'],B_ms=v['B']['ms'],AB_ms=v['AB']['ms'],
                speedup=v['AB']['speedup_vs_orin_projection'],B_speedup=v['B']['speedup_vs_orin_projection'],
                B_vs_none=v['none']['cycles']/v['B']['cycles'],
                AB_vs_none=v['none']['cycles']/v['AB']['cycles'],
                AB_speedup_low=r['baseline']['orin_ms_range'][0]/v['AB']['ms'],
                AB_speedup_high=r['baseline']['orin_ms_range'][1]/v['AB']['ms'],
                class_agreement=q['class_agreement'],mean_cosine=q['mean_cosine'],
                within_tolerance=q['within_tolerance'],rmse=q['rmse'],
                quality_pass=v['AB']['quality_pass'],
                arithmetic_speedup_ceiling=v['AB']['optimistic_arithmetic_speedup_ceiling'],
                removed_flops_fraction=r['pruning']['removed_fraction']))
    if len({case['implementation_revision'] for case in cases})!=1:
        raise ValueError('Mixed implementation revisions; finish the complete matrix before reporting')
    if any(case['config']!=cases[0]['config'] for case in cases):
        raise ValueError('Mixed resource/predictor configurations')
    if records:
        with (directory/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    lines=['# GADRA 九组周期模拟结果','',
        'AGX Orin 为本地 RTX 实测时间/DRAM 与计算 roof 比推算；加速器为事务/tile 周期模拟。每组一个真实完整输入片段，未声称全数据集或整场景测试。',
        '',f"性能实现版本：`{cases[0]['implementation_revision']}`。九组使用同一资源与预测配置。A=语义轨迹/上下文反馈；B=覆盖消费/特征驻留；∅=相同硬件资源关闭 A、B。",
        '', '| 模型 | 数据集 | Orin 推算 ms | 无优化 ms | A ms | B ms | AB ms | AB 加速 | 类别一致 | 余弦 | ε内比例 | 质量门槛 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in records:
        vals=[r['none_ms'],r['A_ms'],r['B_ms'],r['AB_ms']]
        lines.append(f"| {r['model']} | {r['dataset']} | {r['orin_ms']:.2f} | "+' | '.join(f'{x:.2f}' for x in vals)+f" | {r['speedup']:.2f}× | {r['class_agreement']:.2%} | {r['mean_cosine']:.4f} | {r['within_tolerance']:.2%} | {'通过' if r['quality_pass'] else '未通过'} |")
    lines+=['','A 的独立数据集校准不提供分布外硬保证；低质量行不能作为已满足质量的加速结果。B 完整深度输出保持参考值，仍需 RTL/硬件验证。',
             '矩阵峰值、带宽与 SRAM 由每组 JSON 的 config 明确给出，非综合面积/功耗结论。',
             '覆盖前端的 cKDTree 构建/求逆尚未逐指令计时；当前总周期含已建模二次型与链接成本，仍有前端下界假设。',
             '逐组合细节、完整配置、停顿/流量、语义质量和九个跨设备效率情景见对应 JSON。']
    lines+=['','## 机制收益与端到端约束','',
            '| 模型 | 数据集 | B 对 Orin | B 对 ∅ | AB 对 ∅ | AB 对 Orin 效率情景范围 | 删去主算术 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['B_speedup']:.2f}× | {r['B_vs_none']:.3f}× | {r['AB_vs_none']:.3f}× | {r['AB_speedup_low']:.2f}–{r['AB_speedup_high']:.2f}× | {r['removed_flops_fraction']:.2%} |")
    config=cases[0]['config']
    peak=2*config['arrays']*config['array_rows']*config['array_cols']*config['clock_hz']/1e12
    sram=sum(config[name] for name in ('feature_sram_bytes','weight_sram_bytes','output_sram_bytes','control_sram_bytes'))/2**20
    passed=sum(r['quality_pass'] for r in records)
    lines+=['',f"AB 通过质量门槛 {passed}/9 组；5–6× 端到端目标尚未达到。范围仅为所列效率情景，非统计置信区间。低于 1× 表示更慢。",
            f"主配置：{config['arrays']}×{config['array_rows']}×{config['array_cols']} MAC、{config['clock_hz']/1e9:g} GHz，FP16 矩阵峰值 {peak:.3f} TFLOP/s，FP32 {peak/config['fp32_mac_cycles']:.3f} TFLOP/s；SRAM {sram:g} MiB（目录元数据另计），DDR {config['dram_bandwidth_bytes_per_second']/1e9:g} GB/s。",
            'B 的收益须看 B/∅，不能把相同硬件公共执行通路相对 GPU 的收益全部归于覆盖机制。提前结束比例也不能替代实际删除的算术和周期。',
            '质量来自真实检查点向量与原模型输出的比较；此模拟器执行时序/需求，不执行 RTL 浮点数值，未验证硬件舍入误差。',
            '详细解释见 [评测说明](../../docs/evaluation.md)。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
