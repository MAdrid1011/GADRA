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
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    xyz,cov,alpha=geometry(family,dataset,index,geometry_path)
    candidates=min(config['coverage_candidate_limit'],len(xyz))
    _,idx=cKDTree(xyz).query(xyz,k=candidates,workers=4)
    # The midpoint is a common-cover witness only if both true directed
    # quadratic forms pass. Centers/feature similarity never substitute for it.
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    x=torch.as_tensor(xyz,device=dev,dtype=torch.float64)
    c=torch.as_tensor(cov,device=dev,dtype=torch.float64)
    a=torch.as_tensor(alpha,device=dev,dtype=torch.float64)
    inverse=torch.linalg.pinv(c)
    bound=2*torch.log(a/config['coverage_alpha'])
    links=[]
    chunk=config['coverage_candidate_limit']*config['array_rows']
    for begin in range(0,len(x),chunk):
        ids=torch.as_tensor(idx[begin:begin+chunk],device=dev)
        delta=(x[ids]-x[begin:begin+chunk,None])/2
        qa=torch.einsum('nki,nij,nkj->nk',delta,inverse[begin:begin+chunk],delta)
        qb=torch.einsum('nki,nkij,nkj->nk',delta,inverse[ids],delta)
        live=(qa<=bound[begin:begin+chunk,None])&(qb<=bound[ids])&(a[ids]>=config['coverage_alpha'])&(a[begin:begin+chunk,None]>=config['coverage_alpha'])
        # Fixed hardware link budget. Keep nearest witnessed overlaps; no
        # exhaustive graph or unbounded scheduler metadata is assumed.
        rank=torch.cumsum(live,dim=1)
        live&=rank<=config['coverage_link_limit']
        links.append(torch.where(live,ids,-1).cpu().numpy())
    links=np.concatenate(links)
    seen=np.zeros(len(xyz),bool);order=[]
    # Breadth-first front only; finite issue-window scoring happens downstream.
    for seed in range(len(xyz)):
        if seen[seed]:continue
        frontier=[seed];seen[seed]=True
        head=0
        while head<len(frontier):
            node=frontier[head];head+=1;order.append(node)
            for neighbor in links[node]:
                if neighbor>=0 and not seen[neighbor]:seen[neighbor]=True;frontier.append(int(neighbor))
    order=np.asarray(order,dtype=np.int64)
    edges=int((links>=0).sum())
    if archive_path is not None:
        # Preserve the witnessed relation, not just its traversal order. CSR
        # is off-chip metadata; runtime admission has its own finite budget.
        offsets=np.r_[0,np.cumsum((links>=0).sum(axis=1))].astype(np.int64)
        neighbors=links[links>=0].astype(np.int32)
        np.savez_compressed(archive_path,order=order,link_offsets=offsets,link_neighbors=neighbors)
    # 2 quadratic forms: each 3x3 form has 9 multiplies,6 adds plus 3
    # products and2 adds = 20 scalar operations; two forms plus comparisons.
    scalar_ops=len(xyz)*candidates*(2*(9+6+3+2)+2)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return order,dict(rows=len(xyz),candidate_pairs=len(xyz)*candidates,
        links=edges,link_bytes=edges*4+(len(xyz)+1)*8,scalar_ops=scalar_ops,
        scope='Bounded true-ellipsoid common-cover witnesses, not exhaustive overlap rate')
