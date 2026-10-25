"""RTX-measured, stage-wise arithmetic/traffic Orin transfer scenarios."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics

from .workloads import WORKLOADS


def read_kernel_counters(path):
    """Read NCU long/wide CSV exports, requiring all three metrics per kernel."""
    lines=Path(path).read_text().splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
    reader=csv.DictReader(lines[start:])
    metrics=('dram__bytes_read.sum','dram__bytes_write.sum','gpu__time_duration.sum')
    records={};wide_units_validated=False
    for row in reader:
        if not row.get('Kernel Name'):
            if 'Metric Name' not in row and any(row.get(name) for name in metrics):
                expected={name:('nsecond' if name==metrics[2] else 'byte') for name in metrics}
                if any(row.get(name)!=unit for name,unit in expected.items()):
                    raise ValueError('Unexpected wide counter units; export NCU with --print-units base')
                wide_units_validated=True
            continue
        identity=(row.get('Process ID'),row['ID'])
        column=next((k for k in row if k.startswith('thread Domain:')),None)
        matches=re.findall(r'context_stage__([^:\"]+)',row.get(column,'') or '')
        stage=matches[-1] if matches else 'outside_stages'
        record=records.setdefault(identity,dict(stage=stage,values={}))
        if record['stage']!=stage:raise ValueError('Inconsistent stage for one kernel')
        if 'Metric Name' in row:
            name=row['Metric Name']
            if name not in metrics:continue
            if row['Metric Unit']!=('nsecond' if name==metrics[2] else 'byte'):
                raise ValueError('Unexpected counter units')
            if name in record['values']:raise ValueError('Duplicate kernel metric')
            record['values'][name]=float(row['Metric Value'].replace(',',''))
        else:
            # Wide exports include one units row, without a Kernel Name.
            if not wide_units_validated:raise ValueError('Missing wide counter units row')
            record['values']={name:float(row[name].replace(',','')) for name in metrics}
    if not records:raise ValueError('No measured kernels in counter export')
    groups=defaultdict(lambda:dict(bytes=0,ns=0,kernels=0))
    for record in records.values():
        values=record['values']
        if set(values)!=set(metrics):raise ValueError('Incomplete metrics for a measured kernel')
        group=groups[record['stage']]
        group['bytes']+=values[metrics[0]]+values[metrics[1]]
        group['ns']+=values[metrics[2]];group['kernels']+=1
    return dict(groups)


def arithmetic_by_precision(graph):
    """Actual matrix input precision; vector microcode is not tensor FLOPs."""
    costs=defaultdict(lambda:dict(fp16_flops=0,fp32_flops=0))
    for i,event in enumerate(graph.events):
        flops=graph.flops(i)
        if not flops:continue
        width=event['inputs'][0]['element_bytes']
        if width not in (2,4):raise ValueError('Unsupported floating matrix precision')
        costs[event['stage']]['fp16_flops' if width==2 else 'fp32_flops']+=flops
    return dict(costs)


def project_mixed(costs,groups,source_ms,config,counter_source):
    """Stage roof transfer from candidate's own counters and full call time.

    FP16 and FP32 service demands are added conservatively on each device.
    No original-model traffic or original-model timeline enters this function.
    """
    if source_ms<=0:raise ValueError('Complete measured time must be positive')
    if sum(g['ns'] for g in groups.values())<=0:raise ValueError('No positive kernel duration')
    missing=set(costs)-set(groups)
    if missing:raise ValueError(f'Arithmetic stages missing from counters: {sorted(missing)}')
    counter_ns=sum(g['ns'] for g in groups.values());scenarios=[]
    for memory_eff in config['baseline_memory_efficiency_scenarios']:
        for compute_eff in config['baseline_compute_efficiency_scenarios']:
            parts=[]
            for stage,record in groups.items():
                f=costs.get(stage,{});f16=f.get('fp16_flops',0);f32=f.get('fp32_flops',0)
                rtx_compute=f16/config['baseline_rtx_fp16_flops']+f32/config['baseline_rtx_fp32_flops']
                orin_compute=(f16/config['baseline_orin_fp16_flops']+f32/config['baseline_orin_fp32_flops'])/compute_eff
                rtx_roof=max(rtx_compute,record['bytes']/config['baseline_rtx_bandwidth'])
                orin_roof=max(orin_compute,record['bytes']/(config['baseline_orin_bandwidth']*memory_eff))
                rtx_ms=source_ms*record['ns']/counter_ns
                target=rtx_ms*orin_roof/rtx_roof if rtx_roof else rtx_ms
                parts.append(dict(stage=stage,fp16_flops=f16,fp32_flops=f32,bytes=record['bytes'],
                                  rtx_ms=rtx_ms,orin_ms=target,ideal_orin_roof_ms=orin_roof*1000))
            scenarios.append(dict(memory_relative_efficiency=memory_eff,compute_relative_efficiency=compute_eff,
                orin_ms=sum(p['orin_ms'] for p in parts),ideal_orin_roof_ms=sum(p['ideal_orin_roof_ms'] for p in parts),stages=parts))
    reference=next(s for s in scenarios if s['memory_relative_efficiency']==s['compute_relative_efficiency']==1)
    return dict(rtx_measured_ms=source_ms,orin_reference_ms=reference['orin_ms'],
        orin_ms_range=[min(s['orin_ms'] for s in scenarios),max(s['orin_ms'] for s in scenarios)],
        ideal_orin_roof_ms=reference['ideal_orin_roof_ms'],scenarios=scenarios,counter_source=str(counter_source),
        status='projection_only',Orin_measured=False,
        assumptions=['Measured candidate DRAM and same stage arithmetic; actual Orin L2/codegen may differ',
                     'FP16 and FP32 arithmetic service summed, then max with DRAM service',
                     'Uninstrumented complete timeline allocated by measured kernel-stage durations',
                     'Relative efficiency scenarios are sensitivities, not confidence intervals'])


def hardware_stages(model,dataset,index):
    path=WORKLOADS/f'results/memory_flow_v1/dynamic_geometry/{model}_{dataset}_s{index}_original_dynamic.csv'
    lines=path.read_text().splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith('"ID"'))
    reader=csv.DictReader(lines[start:]);units=next(reader)
    if units['dram__bytes_read.sum']!='byte':raise ValueError('Unexpected counter units')
    groups=defaultdict(lambda:dict(bytes=0,ns=0,kernels=0))
    for row in reader:
        if not row.get('Kernel Name'):continue
        column=next(k for k in row if k.startswith('thread Domain:'))
        matches=re.findall(r'context_stage__([^:\"]+)',row[column])
        stage=matches[0] if matches else 'outside_stages'
        groups[stage]['bytes']+=float(row['dram__bytes_read.sum'])+float(row['dram__bytes_write.sum'])
        groups[stage]['ns']+=float(row['gpu__time_duration.sum'])
        groups[stage]['kernels']+=1
    return dict(groups),str(path)


def project(graph,measurement,config):
    # Historical NCU exports are from the established small native fragment.
    # A larger native input requires its own counters and project_mixed().
    for record in (graph.trace, measurement):
        scope=record.get('input_metadata',{}).get('input_scope','chunk-first')
        if scope not in ('chunk-first','native-cache'):
            raise ValueError('Historical counters do not cover this input scope; use its own counters with project_mixed')
    if graph.trace['output_shape']!=measurement['output_shape']:
        raise ValueError('GPU timing and trace have different native output shapes')
    groups,source=hardware_stages(graph.trace['model'],graph.trace['dataset'],graph.trace.get('index',0))
    costs=defaultdict(int)
    for i,e in enumerate(graph.events):costs[e['stage']]+=graph.flops(i)
    fp32=graph.trace['model']=='EmbodiedSplat'
    rp=config['baseline_rtx_fp32_flops' if fp32 else 'baseline_rtx_fp16_flops']
    op=config['baseline_orin_fp32_flops' if fp32 else 'baseline_orin_fp16_flops']
    source_ms=statistics.median(s['gpu_ms'] for s in measurement['samples'])
    counter_ns=sum(g['ns'] for g in groups.values())
    scenarios=[]
    for memory_eff in config['baseline_memory_efficiency_scenarios']:
        for compute_eff in config['baseline_compute_efficiency_scenarios']:
            ms=0;ideal=0;parts=[]
            for stage,record in groups.items():
                f=costs.get(stage,0);b=record['bytes']
                rtx_roof=max(f/rp,b/config['baseline_rtx_bandwidth'])
                orin_roof=max(f/(op*compute_eff),b/(config['baseline_orin_bandwidth']*memory_eff))
                rtx_ms=source_ms*record['ns']/counter_ns
                target=rtx_ms*orin_roof/rtx_roof if rtx_roof else rtx_ms
                ms+=target;ideal+=orin_roof*1000
                parts.append(dict(stage=stage,flops=f,bytes=b,rtx_ms=rtx_ms,orin_ms=target))
            scenarios.append(dict(memory_relative_efficiency=memory_eff,compute_relative_efficiency=compute_eff,
                                  orin_ms=ms,ideal_orin_roof_ms=ideal,stages=parts))
    reference=next(s for s in scenarios if s['memory_relative_efficiency']==s['compute_relative_efficiency']==1)
    return dict(rtx_measured_ms=source_ms,orin_reference_ms=reference['orin_ms'],
                orin_ms_range=[min(s['orin_ms'] for s in scenarios),max(s['orin_ms'] for s in scenarios)],
                ideal_orin_roof_ms=reference['ideal_orin_roof_ms'],scenarios=scenarios,counter_source=source,
                status='projection_only',Orin_measured=False,
                scope='First real spatial fragment, complete 3D semantic encoding; no disk/host input preprocessing',
                assumptions=['Same stage arithmetic and measured DRAM volume; Orin L2/codegen may change actual traffic',
                             'Same relative source efficiency in reference; 0.7/1.0/1.3 efficiency scenarios',
                             'CUDA-event timeline contains host launch gaps; kernel-stage allocation is counter weighted',
                             'No inflated Orin CPU overhead added; ideal Orin roof reported separately'])
