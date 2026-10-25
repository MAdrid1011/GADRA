"""Full-encoding fixed-work timing screens for common execution changes."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from collections import Counter

import numpy as np

from .engine import Simulator
from .experiment import MODELS
from .graph import Graph,rows


def sequential_statistics(encoding,preparation,offset):
    """Sum sequential work, retain one physical capacity and the larger peak."""
    maxima=('integer_control_peak_scratch_bytes','integer_stream_descriptor_bytes','consumer_sort_peak_scratch_bytes',
            'feeder_peak_payload_bytes','feeder_reserved_bytes','attention_peak_staging_bytes',
            'attention_ready_queue_bytes','attention_queue_peak','attention_softmax_register_bytes',
            'vector_ready_queue_bytes','vector_ready_queue_peak',
            'direct_latch_peak_bytes','direct_latch_reserved_bytes','direct_descriptor_bytes','direct_translation_reserved_bytes',
            'dense_accumulator_peak_bytes','input_slice_latch_bytes',
            'input_microprogram_bytes','operand_copy_register_bytes','input_microprogram_peak_bank_entries')
    stats=encoding.copy()
    for name,value in preparation.items():
        previous=stats.get(name,0)
        if name in maxima:stats[name]=max(previous,value)
        elif name!='last_tile_completion':stats[name]=previous+value
    stats['last_tile_completion']=encoding.get('last_tile_completion',0)+offset
    return stats


def compose_preparation_observations(result):
    """Whole-call observers use the same scope as complete cycles/counters.

    Preparation drains before cold encoding starts. The actual phase-local
    histograms can therefore be concatenated exactly; ratios cannot be
    inherited from encoding or averaged. Keep encoding observations explicit.
    This also repairs saved reports without reexecuting or changing timing.
    """
    from .activity import sequential_activity
    prepared=result['preparation'];c=result['config']
    for name in ('execution_intervals','memory_paths','resource_activity'):
        result.setdefault('encoding_'+name,result[name])
    name='copy_admission_observation'
    if name in result or name in prepared:
        if name not in result or name not in prepared:
            raise ValueError('Both phases must observe copy admission')
        encoding=result.setdefault('encoding_'+name,result[name])
        first=prepared[name]
        if first.keys()!=encoding.keys():raise ValueError('Different copy-admission schemas')
        result[name]={key:first[key]+encoding[key] for key in first if key!='scope'}
        result[name]['scope']='Complete sequential preparation + encoding dispatch advances; not E2E removable latency'
    result['execution_intervals']=sequential_activity(prepared['execution_intervals'],
        result['encoding_execution_intervals'],prepared['cycles'])
    a=prepared['memory_paths'];b=result['encoding_memory_paths']
    if a is None and b is None:result['memory_paths']=None
    elif a is None or b is None:raise ValueError('Both phases must observe memory paths')
    else:
        if a.keys()!=b.keys():raise ValueError('Different memory-path schemas')
        result['memory_paths']={name:{k:a[name][k]+b[name][k] for k in a[name]} for name in a}
        for k,total in result['memory_statistics'].items():
            if sum(path[k] for path in result['memory_paths'].values())!=total:
                raise AssertionError('Composed memory paths disagree with full-call counters')
    stats=result['statistics'];cycles=max(1,result['cycles'])
    result['resource_activity']=dict(
        array_issue_fraction=stats.get('array_occupied_cycles',0)/(cycles*c['arrays']),
        vector_issue_fraction=stats.get('vector_cycles',0)/cycles,
        dram_service_fraction=(result['memory_statistics']['dram_read_bytes']+result['memory_statistics']['dram_write_bytes'])*
            c['clock_hz']/c['dram_bandwidth_bytes_per_second']/cycles,
        diagnostic_activity_included=prepared['resource_activity']['diagnostic_activity_included'] and
            result['encoding_resource_activity']['diagnostic_activity_included'])
    result['observation_scope']='Complete sequential preparation + encoding; individual event lists remain phase-local'
    return result


def include_preparation(result,prepared,config):
    """Compose two drained phases without inventing cache/compute overlap.

    Encoding starts cold. Prepared parameters are ordinary input roots and
    are loaded again through encoding's own hierarchy. This common cost is
    paid for every ablation, even when A exits many output rows.
    """
    if 'preparation' in result:raise ValueError('Preparation is already included')
    result['config']=config
    for name in ('cycles','statistics','memory_statistics'):
        value=result[name]
        result['encoding_'+name]=value.copy() if isinstance(value,dict) else value
    result['preparation']=prepared
    result['cycles']+=prepared['cycles'];result['ms']+=prepared['ms']
    if 'reference_flops' in result:
        result['encoding_flops']=result['reference_flops']
        result['reference_flops']+=prepared['reference_flops']
    for name,value in prepared['memory_statistics'].items():
        result['memory_statistics'][name]=result['memory_statistics'].get(name,0)+value
    result['statistics']=sequential_statistics(result['encoding_statistics'],prepared['statistics'],prepared['cycles'])
    result['stage_cycles']['query_preparation']=prepared['cycles']
    if 'operator_totals' in result:
        result['operator_totals'].update(prepared.get('operator_totals',{}))
    result['query_preparation_included']=True
    if result.get('pilot_completed_cycle') is not None:
        result['encoding_pilot_completed_cycle']=result['pilot_completed_cycle']
        result['pilot_completed_cycle']+=prepared['cycles']
    return compose_preparation_observations(result)


def replay_trace(path,config,*,coverage_order=None,coverage_meta=None,coverage_links=None,demand=None):
    """Replay actual selected rows; optionally retain per-operator costs."""
    # This driver executes one complete captured program, with control=None;
    # no deferred feature phase can revisit an earlier SSA version. Keep all
    # ordinary last-reader/observer/alias checks, but permit proven in-place
    # slice writes even when an older export omitted its no-observation flag.
    graph=Graph(path,execution_only=True)
    # A captured causal program has already made its online decisions. Keep
    # the same finite active-demand executor used by run_case; do not silently
    # replay its remaining rows through the no-mechanism issue policy.
    # An explicit False still permits a support-only execution control.
    if demand is None:demand=graph.trace.get('variant') in ('A','A_ancestor')
    outputs=graph.trace.get('preparation_outputs',[graph.events[-1]['outputs'][0]])
    live=graph.closure([(s,np.arange(rows(s))) for s in outputs],release_consumed=True)
    # The completed plan owns its selected row arrays and inheritance ranges.
    # No cycle executor reads closure's byte-granule construction masks. Drop
    # those host-only scratch masks before a long simulation instead of
    # retaining gigabytes of them and forcing independent runs into swap.
    graph.demand.clear()
    if coverage_order is not None:
        expected=graph.trace['output_shape'][0]
        if (coverage_order.shape!=(expected,) or
            not np.array_equal(np.sort(coverage_order),np.arange(expected))):
            raise ValueError('Coverage order must identify exactly this native output row domain')
        native=graph.trace.get('input_metadata')
        if native is not None and (coverage_meta is None or
            coverage_meta.get('input_metadata')!=native or
            coverage_meta.get('native_entry')!=graph.trace.get('entry')):
            raise ValueError('Coverage geometry belongs to another native scene or input scope')
    simulator=Simulator(graph,config,coverage=coverage_order is not None,order=coverage_order,demand=demand,
                        coverage_links=coverage_links)
    simulator.execute(live)
    result=simulator.finish(coverage_meta)
    result['execution_only_graph']=True
    result['active_demand_batches']=bool(demand and config.get('demand_consumer_batches',False))
    result['reference_flops']=sum(graph.flops(i,ids) for i,ids in enumerate(live))
    if config.get('profile_events',False):
        result['events']=simulator.event_profile
        totals={}
        for event in simulator.event_profile:
            name=event['stage']+'/'+event['kind']+'/'+event['op']
            total=totals.setdefault(name,Counter())
            total.update(cycles=event['cycles'],chunks=1,rows=event['rows'],
                         dram_bytes=sum(event['memory'][:2]),sram_bytes=sum(event['memory'][6:]))
            total.update(event['statistics'])
        result['operator_totals']={k:dict(v) for k,v in totals.items()}
    return result,graph.trace


def simulate_trace(trace,configuration,destination,profile=False,*,coverage_path=None):
    """One causal trace, including its separately captured query preparation.

    Preparation finishes before encoding, and uses a cold bounded hierarchy.
    Encoding is also cold: no unmodelled prepared data stays on chip for free.
    """
    trace=Path(trace);config=json.loads(Path(configuration).read_text())
    if profile:config['profile_events']=True
    order=meta=links=None
    if coverage_path is not None:
        coverage_path=Path(coverage_path)
        with np.load(coverage_path) as archive:
            order=archive['order']
            if config.get('coverage_linked_supply',False):
                if not {'link_offsets','link_neighbors'}<=set(archive.files):
                    raise ValueError('Linked supply requires actual captured coverage CSR links')
                links=(archive['link_offsets'],archive['link_neighbors'])
        meta=json.loads(coverage_path.with_suffix('.json').read_text())
        if meta['rows']!=len(order):raise ValueError('Coverage metadata and order domains differ')
    result,metadata=replay_trace(trace,config,coverage_order=order,coverage_meta=meta,coverage_links=links)
    preparation=metadata.get('query_preparation_trace')
    if preparation:
        prepared,_=replay_trace(trace.with_name(preparation),config)
        include_preparation(result,prepared,config)
    else:result['query_preparation_included']=bool(metadata.get('query_preparation_in_trace',False))
    result.update(config=config,trace=str(trace),model=metadata['model'],dataset=metadata['dataset'],
        input_metadata=metadata.get('input_metadata'),
        execution_mechanisms=(['A'] if metadata.get('variant') in ('A','A_ancestor') else [])+(['B'] if order is not None else []),
        coverage_source=str(coverage_path) if coverage_path else None,
        variant=metadata.get('variant'),quality=metadata.get('quality'),
        parent_window=metadata.get('parent_window'),
        active_rows_per_checkpoint=metadata.get('active_rows_per_checkpoint'),
        completed_consumers=metadata.get('completed_consumers'),
        implementation_revision=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip(),
        working_tree_modified=subprocess.run(['git','diff','--quiet']).returncode!=0,
        numba_cache_namespace=os.environ.get('NUMBA_CACHE_DIR'),
        scope='Captured causal full 3D encoding; Orin conversion and software pairing not included')
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(metadata['model'],metadata['dataset'],round(result['ms'],3),'ms',
          'preparation included:',result['query_preparation_included'],flush=True)
    return result


def run_case(model,dataset,configuration,raw,destination):
    config=json.loads(Path(configuration).read_text())
    graph=Graph(Path(raw)/f'{model}_{dataset}_s0.json')
    root=graph.events[-1]['outputs'][0]
    live=graph.closure([(root,np.arange(rows(root)))])
    simulator=Simulator(graph,config)
    simulator.execute(live)
    result=simulator.finish()
    result.update(model=model,dataset=dataset,configuration=configuration,config=config,
        implementation_revision=subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip(),
        working_tree_modified=subprocess.run(['git','diff','--quiet']).returncode!=0,
        reference_flops=sum(graph.flops(i,ids) for i,ids in enumerate(live)),
        scope='First real fragment; full 3D encoding; no A/B; fixed mathematical work')
    output=Path(destination);output.mkdir(parents=True,exist_ok=True)
    name=f'{model}_{dataset}_{Path(configuration).stem}'
    (output/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(name,round(result['ms'],3),'ms',flush=True)
    return result


def run(configurations,dataset,raw,destination,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(run_case,model,dataset,c,raw,destination) for model in MODELS for c in configurations]
        return [job.result() for job in jobs]
