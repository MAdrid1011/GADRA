"""Frozen affine lowerings for Chorus, using the ordinary matrix datapath.

Ancestor sharing preserves the real-valued formula; rank truncation is an
approximation. Both require independent full-call quality and timing. No
final target or trained replacement head enters their execution paths.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time
import types

import torch
from torch import nn
import torch.nn.functional as F

from .workloads import WORKLOADS,load_case


class RankedLinear(nn.Module):
    """Offline, query-independent SVD weights; two ordinary runtime GEMMs.

    The intermediate retains the model's real activation dtype. ``enabled``
    supports the existing alternating native/candidate timing utility.
    """
    def __init__(self,linear,rank):
        super().__init__()
        if not isinstance(linear,nn.Linear) or not 0<rank<=min(linear.weight.shape):
            raise ValueError('Rank requires a native affine and valid dimensions')
        if rank*(linear.in_features+linear.out_features)>=linear.weight.numel():
            raise ValueError('Rank factorization must reduce affine work')
        self.original=linear;self.enabled=True;self.rank=rank
        kwargs=dict(device=linear.weight.device,dtype=linear.weight.dtype)
        self.right=nn.Linear(linear.in_features,rank,bias=False,**kwargs)
        self.left=nn.Linear(rank,linear.out_features,bias=linear.bias is not None,**kwargs)
        with torch.no_grad():
            u,s,v=torch.linalg.svd(linear.weight.float(),full_matrices=False,
                                  driver='gesvd' if linear.weight.is_cuda else None)
            self.right.weight.copy_(v[:rank])
            self.left.weight.copy_(u[:,:rank]*s[:rank])
            if linear.bias is not None:self.left.bias.copy_(linear.bias)
        self.right.requires_grad_(False);self.left.requires_grad_(False)

    def forward(self,source):
        return self.left(self.right(source)) if self.enabled else self.original(source)


@contextmanager
def ranked_language_tail(model,rank):
    """Restore the frozen native output affine even on failed capture."""
    head=model.projectors.lang.mlp_head
    original=head[2];lowered=RankedLinear(original,rank).eval()
    head[2]=lowered
    try:yield lowered
    finally:head[2]=original


def context_banks(point):
    """Consume the native pooling stack and retain actual leaf-to-bank maps."""
    nodes,inverses=[point],[]
    while 'pooling_parent' in point:
        inverses.append(point.pop('pooling_inverse'))
        point=point.pop('pooling_parent');nodes.append(point)
    nodes.reverse();inverses.reverse()
    mapping=torch.arange(len(nodes[0].feat),device=nodes[0].feat.device)
    banks=[(nodes[0].feat,mapping)]
    for node,inverse in zip(nodes[1:],inverses):
        mapping=inverse[mapping];banks.append((node.feat,mapping))
    return nodes[0],banks


class AncestorAffine(nn.Module):
    def __init__(self,projector,channels,projection_dtype,order=None,*,preparation_rows=None):
        super().__init__()
        if not isinstance(projector.input_norm,nn.LayerNorm) or projector.mlp_head is None:
            raise ValueError('Expected LayerNorm followed by an MLP projector')
        first=projector.mlp_head[0];norm=projector.input_norm
        if sum(channels)!=first.in_features:raise ValueError('Ancestor channel partition mismatch')
        self.partials=nn.ModuleList();self.channels=channels;self.eps=norm.eps
        self.enabled=True
        first_weight=first.weight if order is None else first.weight[order]
        first_bias=first.bias if order is None else first.bias[order]
        if preparation_rows is not None:
            if not isinstance(preparation_rows,int) or isinstance(preparation_rows,bool) or preparation_rows<1:
                raise ValueError('Positive preparation row count required')
            # Fold a bounded output-row window through the existing vector
            # multiply/reduce path. All chunks and partial parameter writes
            # are captured, including both reductions and FP16 conversion.
            self.register_buffer('mean_correction',torch.empty(first.out_features,
                device=first.weight.device,dtype=torch.float32))
            self.register_buffer('constant',torch.empty_like(self.mean_correction))
            for width in channels:
                linear=nn.Linear(width,first.out_features,bias=False,device=first.weight.device,dtype=projection_dtype)
                linear.requires_grad_(False);self.partials.append(linear)
            for start in range(0,first.out_features,preparation_rows):
                end=min(start+preparation_rows,first.out_features)
                native=first_weight[start:end].float()
                folded=native*norm.weight.float()[None,:]
                self.mean_correction[start:end].copy_(folded.sum(1))
                self.constant[start:end].copy_((native*norm.bias.float()[None,:]).sum(1)+first_bias[start:end].float())
                begin=0
                for width,linear in zip(channels,self.partials):
                    linear.weight.data[start:end].copy_(folded[:,begin:begin+width]);begin+=width
            return
        folded=first_weight.float()*norm.weight.float()[None,:]
        self.register_buffer('mean_correction',folded.sum(1))
        self.register_buffer('constant',first_weight.float()@norm.bias.float()+first_bias.float())
        begin=0
        for width in channels:
            linear=nn.Linear(width,first.out_features,bias=False,device=first.weight.device,dtype=projection_dtype)
            linear.weight.data.copy_(folded[:,begin:begin+width]);linear.requires_grad_(False)
            self.partials.append(linear);begin+=width

    def prepare(self,banks,inline_fine=False):
        return self.prepare_columns(banks,0,len(self.constant),inline_fine),self.statistics(banks)

    def statistics(self,banks):
        statistics=[]
        for values,mapping in banks:
            variance,mean=torch.var_mean(values.float(),dim=1,unbiased=False)
            statistics.append((mean,variance,mapping))
        return statistics

    def prepare_columns(self,banks,first,last,inline_fine=False):
        """Produce only this real W1 channel stripe; no expanded leaf copies."""
        return [values if inline_fine and index==0 else
                F.linear(values.to(linear.weight.dtype),linear.weight[first:last])
                for index,((values,_),linear) in enumerate(zip(banks,self.partials))]

    def merge(self,projections,statistics,begin,end):
        width=sum(self.channels)
        mean=torch.zeros(end-begin,device=projections[0].device,dtype=torch.float32)
        selected=[]
        for channels,(mu,var,mapping) in zip(self.channels,statistics):
            ids=mapping[begin:end];mu=mu[ids];var=var[ids]
            selected.append((mu,var,ids));mean.add_(mu,alpha=channels/width)
        variance=torch.zeros_like(mean)
        for channels,(mu,var,_) in zip(self.channels,selected):
            variance.add_(var+(mu-mean).square(),alpha=channels/width)
        projected=torch.zeros((end-begin,len(self.constant)),device=mean.device,dtype=torch.float32)
        for value,(_,_,ids) in zip(projections,selected):projected.add_(value[ids].float())
        projected.sub_(mean[:,None]*self.mean_correction)
        projected.mul_(torch.rsqrt(variance+self.eps)[:,None])
        return projected+self.constant


@contextmanager
def ancestor_projector(model,projection_dtype=torch.float32,tile_rows=8192,use_fused=False,inline_fine=False,
                       stripe_columns=None):
    from chorus_runtime import CHANNELS
    projector=model.projectors.lang
    if inline_fine and (not use_fused or projection_dtype!=torch.float16):
        raise ValueError('Inline fine projection requires the FP16 fused kernel')
    if stripe_columns is not None and (stripe_columns<1 or not inline_fine):
        raise ValueError('Striped ancestor projection requires positive width and the inline fine path')
    if hasattr(projector,'ancestor_affine'):raise ValueError('Ancestor projector is already installed')
    original_recovery=model.recovery.forward;original_projector=projector.forward
    projector.ancestor_affine=AncestorAffine(projector,CHANNELS,projection_dtype).eval()
    def recover(self,point):
        if not projector.ancestor_affine.enabled:return original_recovery(point)
        point,banks=context_banks(point)
        point['_gadra_context_banks']=banks
        return point
    def project(self,point):
        if not self.ancestor_affine.enabled:return original_projector(point)
        banks=point['_gadra_context_banks']
        if stripe_columns is not None:
            from .ancestor_triton import merge_gelu
            affine=self.ancestor_affine
            statistics=affine.statistics(banks)
            affine.execution_shape=dict(native_bank_rows=[len(values) for values,_ in banks],
                output_rows=len(point.feat),hidden_columns=len(affine.constant),stripe_columns=stripe_columns,
                coarse_projection_peak_payload_bytes=sum(len(values) for values,_ in banks[1:])*
                    min(stripe_columns,len(affine.constant))*2,
                materialized_hidden_bytes=len(point.feat)*len(affine.constant)*2)
            hidden=torch.empty((len(point.feat),len(affine.constant)),device=point.feat.device,dtype=torch.float16)
            for first in range(0,len(affine.constant),stripe_columns):
                last=min(first+stripe_columns,len(affine.constant))
                projections=affine.prepare_columns(banks,first,last,inline_fine=True)
                for begin in range(0,len(point.feat),tile_rows):
                    end=min(begin+tile_rows,len(point.feat))
                    hidden[begin:end,first:last]=merge_gelu(affine,projections,statistics,begin,end,True,
                        columns=(first,last),projection_columns=(first,last))
            # The materialized hidden tensor is real data, not a free RTL
            # buffer. The trace charges every strided store and second GEMM.
            return torch.cat([self.output_norm(self.mlp_head[2](hidden[begin:begin+tile_rows]))
                              for begin in range(0,len(point.feat),tile_rows)],0)
        projections,statistics=self.ancestor_affine.prepare(banks,inline_fine)
        self.ancestor_affine.execution_shape=dict(native_bank_rows=[len(values) for values,_ in banks],
            output_rows=len(point.feat),hidden_columns=len(self.ancestor_affine.constant),stripe_columns=None,
            coarse_projection_peak_payload_bytes=sum(len(values) for values,_ in banks[1:])*
                len(self.ancestor_affine.constant)*projections[1].element_size(),
            hidden_window_payload_bytes=min(len(point.feat),tile_rows)*len(self.ancestor_affine.constant)*2)
        results=[]
        for begin in range(0,len(point.feat),tile_rows):
            end=min(begin+tile_rows,len(point.feat))
            if use_fused:
                from .ancestor_triton import merge_gelu
                hidden=merge_gelu(self.ancestor_affine,projections,statistics,begin,end,inline_fine)
            else:
                pre=self.ancestor_affine.merge(projections,statistics,begin,end)
                hidden=self.mlp_head[1](pre.to(self.mlp_head[0].weight.dtype))
            results.append(self.output_norm(self.mlp_head[2](hidden)))
        return torch.cat(results,0)
    model.recovery.forward=types.MethodType(recover,model.recovery)
    projector.forward=types.MethodType(project,projector)
    try:yield projector.ancestor_affine
    finally:
        model.recovery.forward=original_recovery;projector.forward=original_projector
        del projector.ancestor_affine


def timed(call,repeats):
    for _ in range(2):output=call()
    samples=[]
    for _ in range(repeats):
        start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize();begin=time.perf_counter();start.record()
        output=call();end.record();end.synchronize()
        samples.append(dict(gpu_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-begin)*1000))
    return output,samples


def paired(call,affine,repeats):
    """Alternate baseline/candidate to avoid comparing different clock epochs."""
    samples={False:[],True:[]}
    for _ in range(2):
        for enabled in (False,True):affine.enabled=enabled;call()
    for repeat in range(repeats):
        for enabled in ((False,True) if repeat%2==0 else (True,False)):
            affine.enabled=enabled
            start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize();begin=time.perf_counter();start.record()
            output=call();end.record();end.synchronize()
            samples[enabled].append(dict(gpu_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-begin)*1000))
            if enabled:result=output
    affine.enabled=True
    return result,samples[False],samples[True]


@torch.inference_mode()
def benchmark(dataset,index,destination,repeats=7):
    from chorus_runtime import texts
    from .numerical import metrics
    from statistics import median
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model,call,context,entry=load_case('Chorus',dataset,index)
    text=texts(dataset).cuda()
    report=dict(model='Chorus',dataset=dataset,index=index,entry=entry,
        status='candidate_operator_factorization',
        scope='First real fragment, complete 3D semantic encoding, CUDA-event measurements',
        mechanism='Ancestor LayerNorm/first-affine sharing; not an A/B paired speedup result')
    try:
        with context:
            reference,samples=timed(call,repeats)
            report['baseline_samples']=samples;report['variants']={}
            for dtype,fused,inline in ((torch.float32,False,False),(torch.float32,True,False),
                                      (torch.float16,False,False),(torch.float16,True,False),
                                      (torch.float16,True,True)):
                with ancestor_projector(model,projection_dtype=dtype,use_fused=fused,inline_fine=inline) as affine:
                    result,baseline_samples,samples=paired(call,affine,repeats)
                    quality=metrics(result,reference,text)
                value=dict(samples=samples,baseline_samples=baseline_samples,quality=quality,
                    measured_speedup=median(s['gpu_ms'] for s in baseline_samples)/median(s['gpu_ms'] for s in samples))
                name=f'{dtype}/'+('inline_fine' if inline else ('fused' if fused else 'torch'))
                report['variants'][name]=value
                print(dataset,name,value['measured_speedup'],quality,flush=True)
        path=Path(destination);path.mkdir(parents=True,exist_ok=True)
        (path/f'Chorus_{dataset}_s{index}_ancestor_affine.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


@torch.inference_mode()
def benchmark_stripes(dataset,index,destination,widths,repeats=7,*,input_scope='chunk-first'):
    """Full native calls, alternating original and common ancestor candidates."""
    from chorus_runtime import texts
    from .numerical import metrics
    from statistics import median
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:
        model,call,context,entry=load_case('Chorus',dataset,index,input_scope=input_scope)
        text=texts(dataset).cuda()
        report=dict(model='Chorus',dataset=dataset,index=index,entry=entry,
            input_metadata=model._gadra_input_scope,status='common_engineering_candidate',
            scope='Complete native first-fragment 3D encoding, original/candidate alternating CUDA events',
            mechanism='Frozen ancestor affine formula and bounded channel projection; no A or coverage B',variants={})
        with context:
            reference,samples=timed(call,repeats)
            report['baseline_samples']=samples
            for width in [None]+list(widths):
                torch.cuda.synchronize();begin=time.perf_counter()
                start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));start.record()
                with ancestor_projector(model,projection_dtype=torch.float16,use_fused=True,inline_fine=True,
                                        stripe_columns=width) as affine:
                    end.record();end.synchronize()
                    preparation=dict(gpu_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-begin)*1000)
                    result,baseline_samples,samples=paired(call,affine,repeats)
                    quality=metrics(result,reference,text)
                    shape=getattr(affine,'execution_shape',None)
                base=median(s['gpu_ms'] for s in baseline_samples)
                candidate=median(s['gpu_ms'] for s in samples)
                name='full_ancestor' if width is None else f'stripe{width}'
                value=dict(preparation=preparation,samples=samples,baseline_samples=baseline_samples,
                    quality=quality,execution_shape=shape,encoding_speedup=base/candidate,
                    one_call_with_preparation_speedup=base/(candidate+preparation['gpu_ms']))
                report['variants'][name]=value
                print(dataset,name,round(base,3),round(candidate,3),round(value['encoding_speedup'],4),quality,flush=True)
        path=Path(destination);path.mkdir(parents=True,exist_ok=True)
        (path/f'Chorus_{dataset}_{input_scope}_stripes.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
