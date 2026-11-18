"""Compare a model's causal execution record with a supplied RTL record.

These functions consume private implementation evidence. They never create an
RTL validation result from a Python replay or from a renamed result file.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .profile import fingerprint


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def capture_digest(raw,tag):
    digest=hashlib.sha256()
    for suffix in ('.json','_maps.pkl','_observations.npz','_geometry.npz'):
        path=Path(raw)/(tag+suffix)
        digest.update(suffix.encode())
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def read_contract(path,config):
    path=Path(path)
    contract=json.loads(path.read_text())
    if contract.get('schema')!='gadra-implementation/v1':
        raise ValueError('Expected a gadra-implementation/v1 resource contract')
    if not contract.get('rtl_revision') or not contract.get('physical_revision'):
        raise ValueError('Name the RTL and physical implementation revisions')
    if contract.get('config_sha256')!=fingerprint(config):
        raise ValueError('The implementation uses another execution configuration')
    expected=dict(vector_lanes=1,scalar_fp_mac_units=1,scalar_iterative_units=1,
                  shares_matrix_array=False)
    for key,value in expected.items():
        if contract.get('resources',{}).get(key)!=value:
            raise ValueError(f'The declared implementation resource {key} differs from the model')
    sources=contract.get('source_files',[])
    if not sources:
        raise ValueError('The contract needs the actual implementation source identities')
    for source in sources:
        if Path(source['path']).suffix not in ('.v','.sv','.scala','.vhd','.vhdl'):
            raise ValueError('Implementation evidence must identify hardware source files')
        if file_digest(path.parent/source['path'])!=source['sha256']:
            raise ValueError('An implementation source differs from its recorded hash')
    if not isinstance(contract.get('operator_mapping'),dict) or not contract['operator_mapping']:
        raise ValueError('Map scalar microprograms to their physical units')
    return contract


def checked_integer(value,name):
    if type(value) is not int or value<0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def audit_signature(events,total_cycles):
    """Check cancellation causality and retain every admitted event-row instance."""
    admissions=Counter()
    cancelled={}
    commits={}
    scalar=[]
    scalar_programs=Counter()
    final_outputs={}
    decisions={}
    for event in events:
        kind=event['kind']
        if kind=='scalar_issue':
            begin=checked_integer(event['begin'],'scalar begin')
            end=checked_integer(event['end'],'scalar end')
            if end<begin or end>total_cycles:
                raise ValueError('A scalar reservation exceeds the execution interval')
            if event.get('resource')!='independent_scalar_sequencer':
                raise ValueError('A scalar reservation names an undeclared parallel resource')
            if not isinstance(event.get('operator'),str) or not event['operator']:
                raise ValueError('Each scalar reservation needs its lowered operator name')
            scalar.append((begin,end))
            scalar_programs[(event['operator'],event.get('domain'),begin,end)]+=1
            continue
        cycle=checked_integer(event.get('cycle',0),'event cycle')
        if cycle>total_cycles:
            raise ValueError('A task event exceeds the execution interval')
        positions=event.get('positions',[])
        if len(positions)!=len(set(positions)) or any(type(i) is not int or i<0 for i in positions):
            raise ValueError('Task rows must be distinct nonnegative integers')
        if kind=='checkpoint_commit':
            depth=checked_integer(event['depth'],'checkpoint depth')
            commits.setdefault(depth,[]).append(cycle)
            for position in positions:
                if position in decisions or position in final_outputs:
                    raise ValueError('A position commits termination twice')
                decisions[position]=(depth,cycle)
        elif kind=='native_issue':
            native=checked_integer(event['event'],'native event')
            for position in positions:
                identity=(native,position)
                if identity in cancelled and cycle>=cancelled[identity]:
                    raise ValueError('A cancelled task is admitted after cancellation')
                admissions[identity]+=1
        elif kind=='task_cancel':
            depth=checked_integer(event['depth'],'cancellation checkpoint')
            decision_cycle=checked_integer(event['decision_cycle'],'cancellation decision cycle')
            if decision_cycle not in commits.get(depth,[]) or cycle<decision_cycle:
                raise ValueError('Cancellation precedes its checkpoint decision commit')
            native=checked_integer(event['event'],'cancelled native event')
            for position in positions:
                identity=(native,position)
                if admissions[identity]:
                    raise ValueError('Cancellation attempts to erase an admitted task')
                if identity in cancelled:
                    raise ValueError('The same pending task is cancelled twice')
                cancelled[identity]=cycle
        elif kind=='native_outputs_commit':
            depth=checked_integer(event['depth'],'native output depth')
            for position in positions:
                if position in decisions or position in final_outputs:
                    raise ValueError('An output is committed more than once')
                final_outputs[position]=(depth,cycle)
    scalar.sort()
    if any(right[0]<left[1] for left,right in zip(scalar,scalar[1:])):
        raise ValueError('Scalar work overlaps on the single physical execution path')
    return dict(admissions=admissions,decisions=decisions,cancelled=set(cancelled),
                scalar_programs=scalar_programs,final_outputs=final_outputs)


def validate_pair(model_record,rtl_path,contract,expected_capture,config):
    rtl=json.loads(Path(rtl_path).read_text())
    if (rtl.get('schema')!='gadra-rtl-trace/v1' or rtl.get('producer')!='verilator'
        or rtl.get('rtl_revision')!=contract['rtl_revision']):
        raise ValueError('A matching Verilator implementation record is required')
    if rtl.get('config_sha256')!=fingerprint(config) or rtl.get('capture_sha256')!=expected_capture:
        raise ValueError('RTL and model records do not use the same capture and configuration')
    if set(rtl.get('variants',{}))!=set(model_record['variants']):
        raise ValueError('RTL and model variant sets differ')
    mapping=contract['operator_mapping']
    allowed={'scalar_fp_mac','scalar_iterative_fp','scalar_control'}
    for name,result in model_record['variants'].items():
        actual=rtl['variants'][name]
        cycles=checked_integer(result['cycles'],'model total cycles')
        if checked_integer(actual.get('cycles'),'RTL total cycles')!=cycles:
            raise ValueError(f'{name}: total cycles differ between RTL and model')
        vector=checked_integer(result['statistics'].get('vector_cycles',0),'model vector work cycles')
        if checked_integer(actual.get('vector_work_cycles'),'RTL vector work cycles')!=vector:
            raise ValueError(f'{name}: scalar work cycles differ between RTL and model')
        first=audit_signature(result['audit_events'],cycles)
        second=audit_signature(actual['audit_events'],cycles)
        if first!=second:
            raise ValueError(f'{name}: admissions, scalar reservations, or commit/cancellation events differ')
        for event in result['audit_events']:
            if event['kind']!='scalar_issue':continue
            implementation=mapping.get(event['operator'])
            if not implementation or not implementation.get('rtl_module'):
                raise ValueError(f'Missing physical operator binding for {event["operator"]}')
            units=implementation.get('units',[])
            if not units or not set(units)<=allowed:
                raise ValueError('An operator uses resources outside the scalar-unit budget')
    for result in model_record['variants'].values():
        result['implementation_validation']='matched_paired_verilator_record'
    return dict(status='matched_paired_verilator_record',rtl_revision=contract['rtl_revision'],
        physical_revision=contract['physical_revision'],capture_sha256=expected_capture,
        config_sha256=fingerprint(config))
