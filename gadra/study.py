"""Fixed scene splits, all-fragment replay, and original-position metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from .profile import validate, fingerprint

MODELS=('SceneSplat','Chorus','EmbodiedSplat')
DATASETS=('scannet','scannetpp','matterport3d')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_manifest(path):
    path=Path(path).resolve();manifest=json.loads(path.read_text())
    if manifest.get('schema')!='gadra-study/v1':raise ValueError('Expected a gadra-study/v1 manifest')
    manifest['_root']=str(path.parent);manifest['_sha256']=digest(path)
    for dataset in DATASETS:
        splits=manifest['splits'][dataset]
        for role,count in (('calibration',10),('tuning',5),('test',10)):
            entries=splits[role]
            if len(entries)!=count or len(set(entries))!=count:
                raise ValueError(f'{dataset}/{role} requires {count} distinct scene IDs')
        groups=[set(splits[r]) for r in ('calibration','tuning','test')]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
            raise ValueError(f'{dataset} calibration, tuning, and test scenes overlap')
    return manifest


def local_path(manifest,path):
    value=Path(path)
    return value if value.is_absolute() else Path(manifest['_root'])/value


def records(manifest,model,role):
    selected=[]
    for record in manifest['models'][model]['fragments']:
        if record['role']!=role:continue
        r=dict(record)
        if r['dataset'] not in DATASETS or r['scene_id'] not in manifest['splits'][r['dataset']][role]:
            raise ValueError('Fragment scene is outside the fixed role split')
        expected_split='validation' if role=='test' else 'training'
        if r.get('source_split')!=expected_split:
            raise ValueError('Scene role must match its official training/validation source split')
        r['raw']=str(local_path(manifest,r['raw']))
        for key in ('orin_timing','scene_labels','scene_indices','readout_manifest'):
            if key in r:r[key]=str(local_path(manifest,r[key]))
        selected.append(r)
    seen={}
    for r in selected:
        key=(r['dataset'],r['scene_id'])
        value=seen.setdefault(key,dict(count=r['fragment_count'],ids=set()))
        if value['count']!=r['fragment_count'] or r['fragment_index'] in value['ids']:
            raise ValueError('Inconsistent fragment count or duplicate fragment')
        value['ids'].add(r['fragment_index'])
    for dataset in DATASETS:
        for scene in manifest['splits'][dataset][role]:
            value=seen.get((dataset,scene))
            if value is None or value['ids']!=set(range(value['count'])):
                raise ValueError(f'Missing full fragment coverage for {model}/{dataset}/{scene}')
    return selected


def readout_identity(path,manifest,model):
    data=json.loads(Path(path).read_text())
    if data.get('model')!=model or data.get('definition')!='l1-cumulative-v1':
        raise ValueError('Frozen diagnostic artifact has another model or definition')
    expected={f'{d}/{s}' for d in DATASETS for s in manifest['splits'][d]['calibration']}
    if set(data['calibration_scenes'])!=expected:
        raise ValueError('Readout fitting must use the fixed 30 calibration scenes')
    for artifact in data['artifacts']:
        target=Path(path).parent/artifact['path']
        if digest(target)!=artifact['sha256']:raise ValueError('Frozen readout artifact hash changed')
    return digest(path)


def calibration_observations(model, entries, config):
    from .graph import Graph,rows
    from .checkpoints import select
    grouped={}
    for r in entries:
        graph=Graph(Path(r['raw'])/(r['tag']+'.json'))
        if graph.trace.get('diagnostic_definition')!='l1-cumulative-v1':
            raise ValueError('Calibration trace predates the manuscript diagnostic definition')
        obs=dict(np.load(Path(r['raw'])/(r['tag']+'_observations.npz')))
        obs,cps=select(model,obs,graph.maps['checkpoints'])
        context=min(cps,key=lambda cp:rows(cp['spec']))['mapping']
        grouped.setdefault((r['dataset'],r['scene_id']),[]).append((obs,context))
    training=[];contexts=[]
    for scene,fragments in sorted(grouped.items()):
        total=sum(o['scores'].shape[1] for o,c in fragments)
        chosen=np.linspace(0,total-1,min(4096,total),dtype=np.int64)
        parts=[];maps=[];offset=0;context_offset=0
        for obs,context in fragments:
            n=obs['scores'].shape[1];selected=chosen[(chosen>=offset)&(chosen<offset+n)]-offset
            parts.append({k:v[:,selected] for k,v in obs.items()
                          if v.ndim>=2 and v.shape[:2]==obs['scores'].shape[:2]})
            maps.append(np.asarray(context)[selected]+context_offset)
            offset+=n;context_offset+=int(np.max(context))+1
        training.append({k:np.concatenate([p[k] for p in parts],axis=1) for k in parts[0]})
        contexts.append(np.concatenate(maps))
    return training,contexts


def merge_scene(parts, labels, class_ids):
    labels=np.asarray(labels,dtype=np.int64);class_ids=np.asarray(class_ids,dtype=np.int64)
    if len(np.unique(class_ids))!=len(class_ids):raise ValueError('Duplicate class ID')
    sums=np.zeros((len(labels),len(class_ids)),np.float64);counts=np.zeros(len(labels),np.int64)
    for vertex_ids,scores in parts:
        vertex_ids=np.asarray(vertex_ids,dtype=np.int64);scores=np.asarray(scores)
        if (scores.shape!=(len(vertex_ids),len(class_ids)) or np.any(vertex_ids<0) or
                np.any(vertex_ids>=len(labels)) or not np.isfinite(scores).all()):
            raise ValueError('Prediction scores do not match original scene indices/classes')
        np.add.at(sums,vertex_ids,scores);np.add.at(counts,vertex_ids,1)
    valid=np.isin(labels,class_ids)
    if np.any(valid&(counts==0)):raise ValueError('Some labeled scene positions have no inference fragment')
    if not valid.any():raise ValueError('The scene has no labels in the selected original class set')
    prediction=class_ids[sums.argmax(1)]
    truth_index={int(c):i for i,c in enumerate(class_ids)}
    gt=np.array([truth_index[int(c)] for c in labels[valid]],np.int64)
    pred=sums[valid].argmax(1)
    confusion=np.bincount(gt*len(class_ids)+pred,minlength=len(class_ids)**2).reshape(len(class_ids),-1)
    return prediction,confusion


def metrics(confusion):
    c=np.asarray(confusion,dtype=np.float64);correct=np.diag(c)
    truth=c.sum(1);union=truth+c.sum(0)-correct
    return dict(miou=float(np.mean(correct[union>0]/union[union>0])*100),
                macc=float(np.mean(correct[truth>0]/truth[truth>0])*100))


def energy_efficiency(speedup, orin_power, gadra_power):
    for measurement in (orin_power,gadra_power):
        if not measurement.get('rail') or not isinstance(measurement.get('interval'),dict):
            raise ValueError('Energy comparison requires the power rail and sampling interval')
        interval=measurement['interval']
        if interval.get('end_ns',0)<=interval.get('start_ns',0):
            raise ValueError('Power sampling needs an ordered start_ns/end_ns interval')
        if measurement.get('excludes_offchip_dram') is not True:
            raise ValueError('The manuscript uses energy excluding off-chip DRAM')
        if not np.isfinite(measurement['mean_watts']) or measurement['mean_watts']<=0:
            raise ValueError('Power must be finite and positive')
    return speedup*orin_power['mean_watts']/gadra_power['mean_watts']


def replay(args):
    from .experiment import run_case
    from .graph import Graph
    from .checkpoints import select
    from .predictor import DemandLUT
    manifest=read_manifest(args.manifest);config=json.loads(args.config.read_text());profile=validate(config,sensitivity=args.sensitivity)
    calibration=records(manifest,args.model,'calibration')
    tests=records(manifest,args.model,args.role)
    identities={readout_identity(r['readout_manifest'],manifest,args.model) for r in calibration+tests}
    if len(identities)!=1:raise ValueError('Calibration and testing must share frozen diagnostic weights')
    training,contexts=calibration_observations(args.model,calibration,config)
    predictor=DemandLUT(training,config,contexts)
    args.output.mkdir(parents=True,exist_ok=True)
    scenes={};runs=[]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for r in tests:
        raw=Path(r['raw']);tag=r['tag'];trace=json.loads((raw/(tag+'.json')).read_text())
        metadata=trace['input_metadata']
        for key in ('scene_id','fragment_index','fragment_count'):
            if metadata.get(key)!=r[key]:raise ValueError(f'Trace and scene manifest disagree on {key}')
        if trace.get('readout_manifest_sha256')!=next(iter(identities)):
            raise ValueError('Trace was captured with different frozen readouts')
        result_dir=args.output/r['dataset']/r['scene_id']/str(r['fragment_index'])
        result_dir.mkdir(parents=True,exist_ok=True)
        reference=json.loads(Path(r['orin_timing']).read_text())
        # A board measurement is read directly for the same scene fragment.
        (result_dir/(tag+'_gpu.json')).write_text(json.dumps(reference,indent=2)+'\n')
        result=Path(run_case(args.model,r['dataset'],args.config,raw,result_dir,
            variant_names=('none','A','B','AB'),case_tag=tag,calibration_records=calibration))
        record=json.loads(result.read_text())
        record.update(manifest_sha256=manifest['_sha256'],source_revision=revision)
        result.write_text(json.dumps(record,indent=2)+'\n')
        graph=Graph(raw/(tag+'.json'));obs,cps=select(args.model,dict(np.load(raw/(tag+'_observations.npz'))),graph.maps['checkpoints'])
        context=min(cps,key=lambda cp:cp['spec']['shape'][0])['mapping']
        stops,_,_=predictor.decide(obs,context)
        scores=obs['scores'];early=scores[stops,np.arange(len(stops))]
        ids=np.load(r['scene_indices'])
        scene=scenes.setdefault((r['dataset'],r['scene_id']),dict(parts=[],full=[],labels=r['scene_labels'],
            orin_ms=0,cycles={v:0 for v in ('none','A','B','AB')},class_ids=manifest['classes'][r['dataset']]))
        scene['parts'].append((ids,early));scene['full'].append((ids,scores[-1]))
        scene['orin_ms']+=record['baseline']['orin_measured_ms']
        for v in scene['cycles']:scene['cycles'][v]+=record['variants'][v]['cycles']
        runs.append(dict(model=args.model,dataset=r['dataset'],scene_id=r['scene_id'],fragment_index=r['fragment_index'],
            orin_ms=record['baseline']['orin_measured_ms'],frequency_hz=config['clock_hz'],
            cycles={v:record['variants'][v]['cycles'] for v in scene['cycles']},
            dram_bytes={v:sum(record['variants'][v]['memory_statistics'].get(k,0) for k in ('dram_read_bytes','dram_write_bytes')) for v in scene['cycles']},
            config_sha256=fingerprint(config),readout_sha256=next(iter(identities)),source_revision=revision))
    aggregate={}
    for (dataset,scene_id),scene in scenes.items():
        labels=np.load(scene['labels']);pred,c=merge_scene(scene['parts'],labels,scene['class_ids'])
        _,full=merge_scene(scene['full'],labels,scene['class_ids'])
        path=args.output/dataset/scene_id;path.mkdir(parents=True,exist_ok=True)
        np.save(path/'prediction.npy',pred)
        (path/'quality.json').write_text(json.dumps(dict(gadra=metrics(c),orin=metrics(full),
            confusion=c.tolist(),orin_confusion=full.tolist()),indent=2)+'\n')
        acc=aggregate.setdefault(dataset,dict(confusion=np.zeros_like(c),full=np.zeros_like(full),orin_ms=0,cycles=0))
        acc['confusion']+=c;acc['full']+=full;acc['orin_ms']+=scene['orin_ms'];acc['cycles']+=scene['cycles']['AB']
    summary=[]
    for dataset,acc in aggregate.items():
        gadra_ms=acc['cycles']/config['clock_hz']*1000
        summary.append(dict(dataset=dataset,model=args.model,orin_ms=acc['orin_ms'],gadra_cycles=acc['cycles'],
            frequency_hz=config['clock_hz'],gadra_ms=gadra_ms,speedup=acc['orin_ms']/gadra_ms,
            gadra_quality=metrics(acc['confusion']),orin_quality=metrics(acc['full'])))
    (args.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(profile=profile,manifest_sha256=manifest['_sha256'],
        source_revision=revision,records=summary),indent=2)+'\n')


def main(model=None):
    parser=argparse.ArgumentParser(description='All-fragment manuscript evaluation with measured Orin timings')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model',choices=MODELS,default=model,required=model is None)
    parser.add_argument('--config',type=Path,default=Path('configs')/((model or 'scenesplat').lower()+'_scannet.json'))
    parser.add_argument('--role',choices=('tuning','test'),default='test')
    parser.add_argument('--sensitivity',action='store_true',help='permit a changed termination threshold or paired residency capacity')
    parser.add_argument('--output',type=Path,default=Path('results/paper'))
    replay(parser.parse_args())

if __name__=='__main__':main()
