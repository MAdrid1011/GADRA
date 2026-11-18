"""Board-measured Orin baselines and Nsight counter parsing."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics
import math



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


def reference_metadata(record):
    """Accept board-measured Orin references for manuscript comparisons."""
    result=dict(record)
    if result.get('Orin_measured') is not True or result.get('status')!='measured':
        raise ValueError('The paper baseline requires a board-measured AGX Orin record')
    if result.get('reference_method')!='orin_cuda_events':
        raise ValueError('The paper baseline requires Orin CUDA-event timing')
    return result


def from_orin_measurement(measurement, *, model=None, dataset=None):
    """Keep device identity, raw milliseconds, power boundary, and run identity."""
    if not measurement.get('measured'):
        raise ValueError('An actual device measurement is required')
    identity=str(measurement.get('device',''))
    if 'orin' not in identity.lower():
        raise ValueError(f'Expected an AGX Orin source device, got {identity!r}')
    for name, expected in (('model',model),('dataset',dataset)):
        if expected is not None and measurement.get(name)!=expected:
            raise ValueError(f'Orin record belongs to another {name}')
    samples=[float(s['gpu_ms']) for s in measurement.get('samples',[])]
    if not samples or any(not math.isfinite(x) or x<=0 for x in samples):
        raise ValueError('Positive raw Orin CUDA-event samples are required')
    elapsed=statistics.median(samples)
    return dict(model=measurement.get('model'),dataset=measurement.get('dataset'),
        orin_reference_ms=elapsed,orin_measured_ms=elapsed,source_measured_ms=elapsed,
        orin_ms_range=[min(samples),max(samples)],status='measured',Orin_measured=True,
        reference_method='orin_cuda_events',device=measurement['device'],
        platform=measurement.get('platform'),samples=measurement['samples'],
        input_metadata=measurement.get('input_metadata',{}),
        scope=measurement.get('scope','Complete 3D semantic encoding'),
        power=measurement.get('power'),run_id=measurement.get('run_id'),
        source_revision=measurement.get('source_revision'),
        quality=measurement.get('quality',{}))


def project(graph,measurement,config):
    """Load the measured baseline corresponding to this exact native trace."""
    if graph.trace['output_shape']!=measurement['output_shape']:
        raise ValueError('Orin timing and trace have different native output shapes')
    expected=graph.trace.get('input_metadata',{})
    actual=measurement.get('input_metadata',{})
    for field in ('scene_id','fragment_index','encoded_rows'):
        if field in expected and expected[field]!=actual.get(field):
            raise ValueError(f'Orin timing and trace disagree on {field}')
    return from_orin_measurement(measurement,model=graph.trace['model'],dataset=graph.trace['dataset'])
