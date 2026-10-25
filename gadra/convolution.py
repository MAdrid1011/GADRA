"""Exact sparse geometry accounting for optional standard convolution lowering."""
from itertools import combinations,product
import json
from pathlib import Path
import pickle

import numpy as np


def ordered_kernel_pairs(pairs,kernel_volume):
    """Kernel-map dictionary iteration order is not the weight offset index."""
    if any(int(k)<0 or int(k)>=kernel_volume for k in pairs):
        raise ValueError('Kernel offset exceeds weight volume')
    return [np.asarray(pairs[k],dtype=np.int64) if k in pairs else np.empty((2,0),dtype=np.int64)
            for k in range(kernel_volume)]


def native_submanifold_map(pairs,counts,rows):
    """Expand Native spconv's implicit center and mirrored count convention."""
    pairs=np.asarray(pairs);counts=np.asarray(counts)
    kernel=len(counts);center=kernel//2
    if kernel%2!=1 or pairs.shape[:2]!=(2,kernel):raise ValueError('Invalid Native submanifold rulebook')
    mapping=np.full((kernel,rows),-1,dtype=np.int32)
    mapping[center]=np.arange(rows,dtype=np.int32)
    for offset in range(kernel):
        if offset==center:continue
        count=int(counts[offset if offset<center else kernel-offset-1])
        source,target=pairs[:,offset,:count]
        if count and (min(source.min(),target.min())<0 or max(source.max(),target.max())>=rows):
            raise ValueError('Invalid Native submanifold pair')
        mapping[offset,target]=source
    return mapping


def transform_candidates(coordinates,mapping):
    """Count actual channel products for direct and tiled F(m,3) choices.

    Missing sites are zero. An absent non-transformed offset contributes zero
    and may be skipped. Holes in a transformed tile still cost products. This
    is a geometry screen, not a latency or numerical-accuracy measurement.
    """
    coord=np.asarray(coordinates,dtype=np.int64)
    mapping=np.asarray(mapping)
    if coord.ndim!=2 or coord.shape[1]!=4 or mapping.shape!=(27,len(coord)):
        raise ValueError('Expected same-coordinate 3x3x3 submanifold convolution')
    edge_counts=(mapping>=0).sum(0)
    direct=int(edge_counts.sum())
    lower=coord.min(0)-8;upper=coord.max(0)+8
    radix=upper-lower+1
    strides=np.array([int(np.prod(radix[i+1:])) for i in range(4)],dtype=np.int64)
    keys=np.sort((coord-lower)@strides)
    result=[]
    for dimensions in (1,2,3):
        for axes in combinations((1,2,3),dimensions):
            other=[axis for axis in (1,2,3) if axis not in axes]
            for tile in (2,4):
                base=coord.copy()
                base[:,axes]=base[:,axes]//tile*tile
                groups,inverse=np.unique(base,axis=0,return_inverse=True)
                group_edges=np.bincount(inverse,weights=edge_counts,minlength=len(groups)).astype(np.int64)
                offsets=np.array(list(product(range(-1,tile+1),repeat=dimensions)),dtype=np.int64)
                transformed=np.zeros(len(groups),dtype=np.int64)
                valid_inputs=np.zeros(len(groups),dtype=np.int64)
                for fixed in product((-1,0,1),repeat=len(other)):
                    candidate=np.repeat(groups[:,None,:],len(offsets),axis=1)
                    candidate[:,:,axes]+=offsets
                    if other:candidate[:,:,other]+=np.array(fixed)
                    encoded=(candidate-lower)@strides
                    where=np.searchsorted(keys,encoded)
                    present=(where<len(keys))&(keys[np.minimum(where,len(keys)-1)]==encoded)
                    transformed+=present.any(1)*(tile+2)**dimensions
                    valid_inputs+=present.sum(1)
                use=transformed<group_edges
                hybrid=int(np.where(use,transformed,group_edges).sum())
                result.append(dict(axes=list(axes),tile=tile,groups=len(groups),
                    transformed_groups=int(use.sum()),direct_channel_products=direct,
                    hybrid_channel_products=hybrid,product_ratio=hybrid/direct,
                    transformed_input_references=int(valid_inputs[use].sum()),
                    note='Products only; transformations, weights, buffers and accuracy not yet validated'))
    return sorted(result,key=lambda item:item['product_ratio'])


def capture_geometry(model,dataset,index,destination):
    import fcntl
    import torch
    import spconv.pytorch as spconv
    from .workloads import WORKLOADS,load_case
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    network,call,context,_=load_case(model,dataset,index)
    collected={};handles=[]
    def hook(name,args):
        value=args[0]
        collected[name]=value.indices.detach()
    for name,module in network.named_modules():
        if isinstance(module,spconv.SubMConv3d) and list(module.kernel_size)==[3,3,3]:
            handles.append(module.register_forward_pre_hook(lambda m,a,name=name:hook(name,a)))
    try:
        with torch.inference_mode(),context:call()
        # Geometry collection is untimed. One transfer per distinct coordinate
        # storage, not per Gaussian or repeated convolution at the same scale.
        arrays={};names={};storages={}
        for name,value in collected.items():
            identity=int(value.untyped_storage()._cdata)
            if identity not in storages:
                coordinate_name=f'coordinates_{len(arrays)}';storages[identity]=coordinate_name
                arrays[coordinate_name]=value.cpu().numpy()
            names[name]=storages[identity]
        path=Path(destination);path.mkdir(parents=True,exist_ok=True)
        tag=f'{model}_{dataset}_s{index}'
        np.savez_compressed(path/f'{tag}_conv_geometry.npz',**arrays)
        (path/f'{tag}_conv_geometry.json').write_text(json.dumps(names,indent=2)+'\n')
    finally:
        for handle in handles:handle.remove()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def repair_rulebooks(model,dataset,index,raw_directory):
    """Repair only proven-missing Native edges using an actual model call."""
    import fcntl
    import torch
    import spconv.pytorch as spconv
    from .workloads import WORKLOADS,load_case
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    network,call,context,_=load_case(model,dataset,index)
    corrected={};handles=[]
    def hook(name,mod,args,out):
        info=out.indice_dict[mod.indice_key]
        if hasattr(info,'pair_fwd'):return
        corrected[name]=native_submanifold_map(info.indice_pairs.cpu().numpy(),
            info.indice_pair_num.cpu().numpy(),len(out.features))
    for name,module in network.named_modules():
        if isinstance(module,spconv.SubMConv3d):
            handles.append(module.register_forward_hook(lambda m,a,o,name=name:hook(name,m,a,o)))
    try:
        with torch.inference_mode(),context:call()
        path=Path(raw_directory);tag=f'{model}_{dataset}_s{index}'
        portable=path/f'{tag}_maps.pkl'
        maps=pickle.loads(portable.read_bytes())
        native_path=path/f'{tag}_maps.pt'
        native=torch.load(native_path,map_location='cpu',weights_only=False)
        changes=[]
        for name,mapping in corrected.items():
            old=maps['maps'][name]['mapping']
            if old.shape!=mapping.shape:raise ValueError(('Changed geometry shape',name))
            # Previously explicit entries must still identify the same input;
            # this operation fills missing real edges, never changes them.
            if not np.array_equal(old[old>=0],mapping[old>=0]):raise ValueError(('Changed existing edge',name))
            changes.append(dict(name=name,old_edges=int((old>=0).sum()),new_edges=int((mapping>=0).sum())))
            maps['maps'][name]['mapping']=mapping
            maps['maps'][name]['native_submanifold_expanded']=True
            native['maps'][name]['mapping']=torch.from_numpy(mapping)
            native['maps'][name]['native_submanifold_expanded']=True
        portable.write_bytes(pickle.dumps(maps,protocol=4))
        torch.save(native,native_path)
        (path/f'{tag}_rulebook_repair.json').write_text(json.dumps(changes,indent=2)+'\n')
        return changes
    finally:
        for handle in handles:handle.remove()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
