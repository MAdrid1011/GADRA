"""Prepare every manifest fragment and fit frozen diagnostic readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .study import MODELS, DATASETS, read_manifest, records, local_path, digest


def fit_affine(source,target):
    """Ordinary least squares for a bias-augmented calibration source."""
    source=np.asarray(source,np.float32);target=np.asarray(target,np.float32)
    if source.ndim!=2 or target.ndim!=2 or len(source)!=len(target) or not len(source):
        raise ValueError('Affine fitting needs aligned nonempty matrices')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Affine fitting needs finite features and targets')
    return np.linalg.lstsq(source,target,rcond=None)[0].astype(np.float32)


def fit_readouts(manifest,model,output):
    entries=records(manifest,model,'calibration');scenes={}
    for r in entries:
        arrays=dict(np.load(Path(r['raw'])/(r['tag']+'_features.npz')))
        scenes.setdefault((r['dataset'],r['scene_id']),[]).append(arrays)
    per_checkpoint={};targets=[];head=None
    for scene,parts in sorted(scenes.items()):
        sizes=[len(a['final']) for a in parts];total=sum(sizes)
        selected=np.linspace(0,total-1,min(4096,total),dtype=np.int64);offset=0
        for part,n in zip(parts,sizes):
            ids=selected[(selected>=offset)&(selected<offset+n)]-offset;offset+=n
            target=part['teacher_latent'] if model=='EmbodiedSplat' else part['final']
            targets.append(target[ids])
            count=sum(key.startswith('features_') for key in part)
            for d in range(count):
                current=part[f'features_{d}'][part[f'mapping_{d}'][ids]]
                if model=='Chorus':
                    prefix=[part[f'features_{j}'][part[f'mapping_{j}'][ids]] for j in range(d)]
                    current=np.concatenate(prefix+[current],axis=1)
                if model!='EmbodiedSplat':
                    current=current/np.maximum(np.linalg.norm(current,axis=1,keepdims=True),np.finfo(np.float32).tiny)
                current=np.pad(current,((0,0),(0,1)),constant_values=0)
                current[:,-1]=1
                per_checkpoint.setdefault(d,[]).append(current.astype(np.float32))
            if model=='EmbodiedSplat':
                value=(part['language_weight'],part['language_bias'])
                if head is not None and any(not np.array_equal(a,b) for a,b in zip(head,value)):
                    raise ValueError('The frozen native semantic head changed across calibration scenes')
                head=value
    target=np.concatenate(targets);expected=64 if model=='EmbodiedSplat' else {'SceneSplat':768,'Chorus':1152}[model]
    if target.shape[1]!=expected:raise ValueError('Native target width differs from the manuscript readout')
    weights={}
    for d,parts in per_checkpoint.items():
        source=np.concatenate(parts)
        weights[f'weight_{d}']=fit_affine(source,target)
    if head is not None:weights.update(language_weight=head[0],language_bias=head[1])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'weights.npz';np.savez_compressed(path,**weights)
    description=dict(model=model,definition='l1-cumulative-v1',fit='affine_least_squares',
        backbone='frozen',calibration_scenes=[f'{d}/{s}' for d,s in sorted(scenes)],
        max_positions_per_scene_checkpoint=4096,artifacts=[dict(path=path.name,sha256=digest(path))])
    (output/'readouts.json').write_text(json.dumps(description,indent=2)+'\n')


def main(model=None):
    parser=argparse.ArgumentParser(description='Manifest-driven paper input preparation')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model',choices=MODELS,default=model,required=model is None)
    parser.add_argument('--phase',choices=('features','fit','capture','orin'),required=True)
    parser.add_argument('--role',choices=('calibration','tuning','test'),default='test')
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args();manifest=read_manifest(args.manifest)
    if args.phase=='fit':
        fit_readouts(manifest,args.model,local_path(manifest,manifest['models'][args.model]['readout_dir']))
        return
    role='calibration' if args.phase=='features' else args.role
    selected=records(manifest,args.model,role)
    for r in selected:
        runtime_role={'test':'evaluation','calibration':'calibration','tuning':'tuning'}[role]
        raw=Path(r['raw']);raw.mkdir(parents=True,exist_ok=True)
        index=r['source_index'];fragment=r['fragment_index']
        expected=f'{args.model}_{r["dataset"]}_s{index}_f{fragment}'
        if r['tag']!=expected:raise ValueError(f'Trace tag must be {expected}')
        if args.phase=='orin':
            from .workloads import gpu_baseline
            result=gpu_baseline(args.model,r['dataset'],index,args.repeats,
                input_scope='scene-all',role=runtime_role,fragment_index=fragment,scene_id=r['scene_id'])
            path=Path(r['orin_timing']);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+'\n');continue
        if args.phase=='capture' and not r.get('readout_manifest'):
            raise ValueError('Diagnostic capture requires its fixed calibration readout manifest')
        from .capture import capture
        capture(args.model,r['dataset'],index,raw,input_scope='scene-all',role=runtime_role,
            fragment_index=fragment,scene_id=r['scene_id'],features_only=args.phase=='features',
            readout_manifest=r.get('readout_manifest') if args.phase=='capture' else None)
        if args.phase=='capture':
            from .coverage import build_order
            config=json.loads((Path(__file__).resolve().parents[1]/'configs'/
                f'{args.model.lower()}_scannet.json').read_text())
            _,meta=build_order(args.model,r['dataset'],index,config,
                geometry_path=raw/(r['tag']+'_geometry.npz'),archive_path=raw/(r['tag']+'_coverage.npz'))
            (raw/(r['tag']+'_coverage.json')).write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
