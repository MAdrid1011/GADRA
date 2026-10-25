"""Explicit mixed boundaries and exact duplicate-row head sharing.

The default changes only existing submanifold children. Extended mode also
changes affine kernels and commutes the row-local head with Gaussian expansion.
Input-dependent normalization and transposed convolution remain unchanged;
every activation cast is executed and captured.
"""
from contextlib import contextmanager
import copy
import types


@contextmanager
def submanifold_fp16(model,extended=False,wide_kernel=False):
    if wide_kernel and not extended:
        raise ValueError('Wide-kernel precision requires the extended candidate')
    import torch
    from torch import nn
    import spconv.pytorch as spconv

    state=types.SimpleNamespace(enabled=True,calls=0)

    class Boundary(nn.Module):
        def __init__(self,original):
            super().__init__()
            self.original=original
            self.half_kernel=copy.deepcopy(original).half().eval()

        def forward(self,value,*args,**kwargs):
            if not state.enabled:return self.original(value,*args,**kwargs)
            state.calls+=1
            result=self.half_kernel(value.replace_feature(value.features.half()),*args,**kwargs)
            return result.replace_feature(result.features.to(value.features.dtype))

    restored=[];extra=[];linear_restored=[];dense_sparse=[]
    cuda_include_state=None;cuda_include_changed=False
    original_forward=model.forward
    try:
        if extended:
            # Debian installs CUDA headers in /usr/include rather than
            # cumm's hard-coded /usr/local/cuda/include. Point JIT compilation
            # at the installed toolkit; compile time is outside encoding.
            from pathlib import Path
            import shutil
            import cumm.common
            toolkit=shutil.which('nvcc')
            if toolkit:
                include=Path(toolkit).resolve().parents[1]/'include'
                if (include/'cuda/std/cassert').exists():
                    cuda_include_state=cumm.common._CACHED_CUDA_INCLUDE_LIB
                    cumm.common._CACHED_CUDA_INCLUDE_LIB=(include,include.parent/'lib')
                    cuda_include_changed=True
            # Reuse the existing coordinate-template/ME adapter rather than
            # implementing a second sparse runtime. Only its frozen weight
            # conversion and kernel-size metadata differ.
            import MinkowskiEngine as ME
            template=next(module.forward.__func__ for module in model.modules()
                          if isinstance(getattr(module,'equivalent_spconv',None),spconv.SubMConv3d))
            for module in list(model.modules()):
                if not isinstance(module,ME.MinkowskiConvolution) or hasattr(module,'equivalent_spconv'):continue
                gen=module.kernel_generator
                if not wide_kernel or gen.kernel_size!=[5,5,5] or gen.kernel_stride!=[1,1,1]:continue
                if len(set(gen.kernel_dilation))!=1 or gen.region_type!=ME.RegionType.HYPER_CUBE:
                    raise ValueError('Wide equivalent kernel requires isotropic hypercube geometry')
                dilation=gen.kernel_dilation[0]
                layer=spconv.SubMConv3d(module.in_channels,module.out_channels,5,
                    dilation=dilation,bias=module.bias is not None,indice_key=f'k5_d{dilation}',
                    algo=spconv.ConvAlgo.MaskImplicitGemm).to(module.kernel.device)
                with torch.no_grad():
                    converted=module.kernel.reshape(5,5,5,module.in_channels,module.out_channels)
                    layer.weight.copy_(converted.permute(4,2,1,0,3).contiguous())
                    if module.bias is not None:layer.bias.copy_(module.bias.flatten())
                defaults=list(template.__defaults__)
                names=template.__code__.co_varnames[:template.__code__.co_argcount]
                defaults[names.index('layer')-(len(names)-len(defaults))]=layer
                rebound=types.FunctionType(template.__code__,template.__globals__,template.__name__,tuple(defaults),template.__closure__)
                extra.append((module,module.forward))
                module.equivalent_spconv=layer;module.forward=types.MethodType(rebound,module)
            class LinearBoundary(Boundary):
                def forward(self,value):
                    if not state.enabled:return self.original(value)
                    state.calls+=1
                    return self.half_kernel(value.half()).to(value.dtype)
            for parent in list(model.modules()):
                for name,child in list(parent.named_children()):
                    if isinstance(child,nn.Linear):
                        linear_restored.append((parent,name,child))
                        setattr(parent,name,LinearBoundary(child))
            for module in list(model.modules()):
                if not isinstance(module,ME.MinkowskiConvolution) or not module.use_mm:continue
                layer=nn.Linear(module.in_channels,module.out_channels,bias=module.bias is not None,
                                device=module.kernel.device,dtype=torch.float16)
                with torch.no_grad():
                    layer.weight.copy_(module.kernel.T)
                    if module.bias is not None:layer.bias.copy_(module.bias.flatten())
                previous=module.forward
                def sparse_linear(self,value,coordinates=None,layer=layer,previous=previous):
                    if not state.enabled or coordinates is not None:return previous(value,coordinates)
                    state.calls+=1
                    result=layer(value.F.half()).to(value.F.dtype)
                    return ME.SparseTensor(features=result,coordinate_map_key=value.coordinate_map_key,
                                           coordinate_manager=value.coordinate_manager)
                dense_sparse.append((module,previous))
                module.equivalent_linear=layer;module.forward=types.MethodType(sparse_linear,module)
            def shared_head(self,coordinates,features,accumulated=None,inverse=None):
                if not state.enabled:return original_forward(coordinates,features,accumulated,inverse)
                # The frozen ReLU/affine/normalization acts independently on
                # each row, so apply it once per original sparse row before
                # restoring Gaussian duplicates. Input-dependent BN stays in
                # its original position inside the unchanged U-Net.
                result=original_forward(coordinates,features,accumulated,None)
                return result if inverse is None else result[inverse]
            model.forward=types.MethodType(shared_head,model)
        for module in list(model.modules()):
            layer=getattr(module,'equivalent_spconv',None)
            if not isinstance(layer,spconv.SubMConv3d):continue
            original=module.forward
            function=original.__func__
            names=function.__code__.co_varnames[:function.__code__.co_argcount]
            defaults=list(function.__defaults__ or ())
            if 'layer' not in names:raise ValueError('Equivalent kernel is not bound by this runtime')
            position=names.index('layer')-(len(names)-len(defaults))
            if position<0 or defaults[position] is not layer:
                raise ValueError('Equivalent kernel binding differs from the registered layer')
            boundary=Boundary(layer)
            defaults[position]=boundary
            rebound=types.FunctionType(function.__code__,function.__globals__,function.__name__,
                                       tuple(defaults),function.__closure__)
            restored.append((module,original,layer))
            module.equivalent_spconv=boundary
            module.forward=types.MethodType(rebound,module)
        # A disabled precision context must execute the real original ME k5,
        # not the much slower converted FP32 SpConv implementation.
        for module,previous in extra:
            converted=module.forward
            def optional_wide(self,*args,previous=previous,converted=converted,**kwargs):
                return (converted if state.enabled else previous)(*args,**kwargs)
            module.forward=types.MethodType(optional_wide,module)
        if not restored:raise ValueError('No equivalent submanifold kernels to change precision')
        yield state
    finally:
        for module,forward,layer in restored:
            module.forward=forward
            module.equivalent_spconv=layer
        for module,forward in extra:
            module.forward=forward
            del module.equivalent_spconv
        for parent,name,child in linear_restored:setattr(parent,name,child)
        for module,forward in dense_sparse:
            module.forward=forward
            del module.equivalent_linear
        model.forward=original_forward
        if cuda_include_changed:
            cumm.common._CACHED_CUDA_INCLUDE_LIB=cuda_include_state


def benchmark(dataset,destination,repeats=7,extended=False,wide_kernel=False):
    """Alternate complete FP32/mixed encodings; immutable weight conversion is offline."""
    import fcntl
    import json
    from pathlib import Path
    from statistics import median
    import torch
    from .workloads import WORKLOADS,load_case
    from .factorized import paired
    from .numerical import metrics

    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model,call,context,entry=load_case('EmbodiedSplat',dataset,0)
        text=torch.load(WORKLOADS/'results/embodiedsplat_native_measurement/texts.pt',
                        weights_only=False)[dataset]['features'].cuda()
        with torch.inference_mode(),context:
            reference=call().clone()
            with submanifold_fp16(model,extended,wide_kernel) as state:
                actual,baseline,samples=paired(call,state,repeats)
                quality=metrics(actual,reference,text)
        result=dict(model='EmbodiedSplat',dataset=dataset,entry=entry,
            variant=('mixed_precision_extended_wide' if wide_kernel else 'mixed_precision_extended') if extended else 'mixed_precision_submanifold',extended=extended,wide_kernel=wide_kernel,quality=quality,baseline_samples=baseline,samples=samples,
            measured_speedup=median(s['gpu_ms'] for s in baseline)/median(s['gpu_ms'] for s in samples),
            scope='Complete 3D encoding, actual FP16 kernel boundaries; static weight conversion outside encoding',
            status='Common precision candidate, not A/B or hardware paired')
        path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(dataset,result['measured_speedup'],quality,flush=True)
        return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def observe(dataset,destination,counters=False,extended=False,wide_kernel=False):
    """Capture actual child precision/casts, or profile the same complete call."""
    import fcntl
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from .workloads import WORKLOADS,load_case
    from .capture import trace_type,dump_trace
    from .numerical import metrics

    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model,call,context,entry=load_case('EmbodiedSplat',dataset,0)
        text=torch.load(WORKLOADS/'results/embodiedsplat_native_measurement/texts.pt',
                        weights_only=False)[dataset]['features'].cuda()
        with torch.inference_mode(),context:
            reference=call().clone()
            with submanifold_fp16(model,extended,wide_kernel):
                for _ in range(2):expected=call().clone()
                control=call().clone()
                control_repeat_error=float((control.float()-expected.float()).abs().max())
                control_quality=metrics(control,reference,text)
                if counters:
                    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
                    try:
                        # One range matches the one-stage trace in this screen.
                        with torch.cuda.nvtx.range('context_stage__outside_stages'):
                            actual=call();torch.cuda.synchronize()
                    finally:torch.cuda.cudart().cudaProfilerStop()
                else:
                    trace=trace_type('EmbodiedSplat')(model,None,None)
                    trace.attach()
                    try:
                        with trace:actual=call()
                    finally:trace.close()
                quality=metrics(actual,reference,text)
                repeat_error=float((actual.float()-expected.float()).abs().max())
                if not quality['passes_thresholds'] or not control_quality['passes_thresholds']:
                    raise ValueError('Mixed-precision control/capture failed quality')
                metadata=dict(model='EmbodiedSplat',dataset=dataset,index=0,entry=entry,
                    variant=('mixed_precision_extended_wide' if wide_kernel else 'mixed_precision_extended') if extended else 'mixed_precision_submanifold',extended=extended,wide_kernel=wide_kernel,quality=quality,
                    scope='Actual complete mixed-precision 3D encoding; static weights prepared offline',
                    candidate_repeat_max_abs_error=repeat_error,
                    control_repeat_max_abs_error=control_repeat_error,
                    control_quality=control_quality,
                    repeat_limit='Native mixed-precision execution is not assumed bitwise deterministic; both paths must pass original FP32 quality')
                if counters:
                    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(metadata,indent=2)+'\n')
                else:
                    cp=dict(name='final',spec=trace.spec(actual),mapping=np.arange(len(actual)),
                            after_event=len(trace.events)-1)
                    dump_trace(trace,actual,destination,f'EmbodiedSplat_{dataset}_s0',metadata,[cp])
        print(dataset,'counters' if counters else 'trace',quality,flush=True)
        return metadata
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def project(trace_path,timing_path,counter_path,counter_metadata,configuration,destination):
    """Orin projection of this candidate's own complete software execution."""
    import json
    from pathlib import Path
    from statistics import median
    from .graph import Graph
    from .baseline import arithmetic_by_precision,read_kernel_counters,project_mixed

    graph=Graph(trace_path);timing=json.loads(Path(timing_path).read_text())
    counters=json.loads(Path(counter_metadata).read_text());meta=graph.trace
    for source in (timing,counters):
        if any(source.get(k)!=meta.get(k) for k in ('model','dataset','variant','extended','wide_kernel')):
            raise ValueError('Different mixed-precision software case')
        if not source['quality']['passes_thresholds']:raise ValueError('Software candidate failed quality')
    if not meta['quality']['passes_thresholds']:raise ValueError('Captured software failed quality')
    config=json.loads(Path(configuration).read_text())
    result=project_mixed(arithmetic_by_precision(graph),read_kernel_counters(counter_path),
                         median(s['gpu_ms'] for s in timing['samples']),config,counter_path)
    result.update(model=meta['model'],dataset=meta['dataset'],variant=meta['variant'],
                  quality=timing['quality'],trace_source=str(trace_path),timing_source=str(timing_path),
                  counter_metadata_source=str(counter_metadata),
                  scope=meta['scope'],comparison_note='Mixed-precision candidate-specific software projection, not A/B')
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(meta['dataset'],'same-candidate Orin projection',result['orin_reference_ms'],'ms',flush=True)
    return result
