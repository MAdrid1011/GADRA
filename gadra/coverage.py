"""Finite geometric overlap links and legal same-operator ready ordering."""
from pathlib import Path
import fcntl
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from .workloads import WORKLOADS


def geometry(family,dataset,index,geometry_path=None):
    if family=='EmbodiedSplat':
        if geometry_path is not None:
            g=np.load(geometry_path)
            return g['means'],g['covariances'],g['opacities']
        p=WORKLOADS/f'results/xklr_v1/geometry/{family}_{dataset}_s{index}.pt'
        if not p.exists():
            p=WORKLOADS/f'results/embodiedsplat_native_measurement/geometry/{family}_{dataset}_s{index}.pt'
        g=torch.load(p,map_location='cpu',weights_only=False)
        return g['means'].numpy(),g['covariances'].numpy(),g['opacities'].numpy()
    if geometry_path is None:
        raise ValueError('SceneSplat and Chorus coverage require captured native geometry')
    g=np.load(geometry_path)
    xyz,feat=g['coord'],g['feat']
    rotation=Rotation.from_quat(feat[:,[5,6,7,4]]).as_matrix()
    cov=(rotation*feat[:,None,8:11]**2)@rotation.transpose(0,2,1)
    return xyz,cov,feat[:,3]


def build_order(family,dataset,index,config,*,geometry_path=None,archive_path=None):
    """Bounded scale-normalized priorities, followed by dependency-ready issue."""
    xyz,cov,_=geometry(family,dataset,index,geometry_path)
    xyz=np.asarray(xyz,dtype=np.float64)
    if not len(xyz):raise ValueError('Coverage ordering requires a nonempty native input')
    squared_scales=np.trace(cov,axis1=1,axis2=2)
    candidates=min(config['coverage_candidate_limit'],len(xyz))
    _,neighbors=cKDTree(xyz).query(xyz,k=candidates,workers=4)
    neighbors=np.asarray(neighbors).reshape(len(xyz),candidates)
    denominator=squared_scales[:,None]+squared_scales[neighbors]
    distances=((xyz[:,None]-xyz[neighbors])**2).sum(2)/np.maximum(denominator,np.finfo(np.float64).tiny)
    priority=np.argsort(distances,axis=1,kind='stable')
    links=np.take_along_axis(neighbors,priority,axis=1)[:,:config['coverage_link_limit']]
    seen=np.zeros(len(xyz),bool);order=[]
    for seed in range(len(xyz)):
        if seen[seed]:continue
        frontier=[seed];seen[seed]=True
        for node in frontier:
            order.append(node)
            for neighbor in links[node]:
                if not seen[neighbor]:seen[neighbor]=True;frontier.append(int(neighbor))
    order=np.asarray(order,dtype=np.int64)
    offsets=np.arange(len(xyz)+1,dtype=np.int64)*links.shape[1]
    if archive_path is not None:
        np.savez_compressed(archive_path,order=order,link_offsets=offsets,
            link_neighbors=links.reshape(-1).astype(np.int32))
    return order,dict(rows=len(xyz),candidate_pairs=len(xyz)*candidates,
        links=int(links.size),link_bytes=links.size*4+offsets.nbytes,
        scalar_ops=int(5*len(xyz)+len(xyz)*candidates*(10+np.ceil(np.log2(max(2,candidates))))),
        ordering='squared_center_distance_over_sum_squared_scales',
        scope='Scale priorities. Original network dependencies identify shared inputs.')
