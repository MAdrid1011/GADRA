"""Functional replay from initial tensors; no recorded feature outputs as inputs."""
from dataclasses import dataclass
from contextlib import ExitStack
import fcntl
import gc
import json
import math
from pathlib import Path
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten,tree_map

from .capture import trace_type
from .workloads import WORKLOADS,load_case


@dataclass(frozen=True)
class Ref:
    number:int


def view_spec(value):
    return dict(dtype=value.dtype,device=value.device,shape=tuple(value.shape),
                stride=tuple(value.stride()),offset=value.storage_offset())


def storage_view(base,spec):
    return torch.empty(0,dtype=spec['dtype'],device=spec['device']).set_(
        base.untyped_storage(),spec['offset'],spec['shape'],spec['stride'])


def recorder_type(family,*,retain_allocations=True):
    class Recorder(trace_type(family)):
        def __init__(self,model,official,allowed):
            super().__init__(model,None,official)
            self.program=[];self.roots={};self.objects={};self.owners={};self.held_storages={}
            self.next_ref=0;self.carriers={};self.allowed=allowed|self.parameter_keys
            self.owner_specs={};self.root_parameters=set()
            self.rng_cpu=None;self.rng_cuda=None
            if not retain_allocations:
                from .capture import WeakStorageOwners
                self.storage_handles=WeakStorageOwners()
                self.held_storages=WeakStorageOwners()

        def __enter__(self):
            self.original_manual_seed=torch.manual_seed
            def manual_seed(seed):
                # Python seed resets are not ATen operations. Parameter
                # preparation can consume RNG before the encoder resets it.
                self.program.append(dict(kind='rng_reset',seed=int(seed)))
                return self.original_manual_seed(seed)
            torch.manual_seed=manual_seed
            try:return super().__enter__()
            except BaseException:
                torch.manual_seed=self.original_manual_seed
                raise

        def __exit__(self,*args):
            try:return super().__exit__(*args)
            finally:torch.manual_seed=self.original_manual_seed

        def encode(self,tree,produced=False):
            def tensor(value):
                if not isinstance(value,torch.Tensor):return value
                previous=self.objects.get(id(value))
                if previous is not None and previous[0]() is value:return previous[1]
                reference=Ref(self.next_ref);self.next_ref+=1
                self.objects[id(value)]=(weakref.ref(value),reference)
                identity=(str(value.device),int(value.untyped_storage()._cdata))
                # Strong owners retain data; weak StorageImpl owners retain
                # identity only. Both prevent _cdata reuse during capture.
                self.spec(value)
                if identity not in self.held_storages:
                    self.held_storages[identity]=value.untyped_storage()
                if not produced:
                    if identity in self.owners:
                        self.program.append(dict(kind='alias',base=self.owners[identity],
                                                 base_spec=self.owner_specs[identity],
                                                 spec=view_spec(value),output=reference))
                    else:
                        if value.is_floating_point() and value.numel()>1 and self.key(value) not in self.allowed:
                            raise ValueError(f'Untracked floating input: {tuple(value.shape)}')
                        base=torch.empty(0,dtype=torch.uint8,device=value.device).set_(
                            value.untyped_storage(),0,(value.untyped_storage().nbytes(),),(1,)).clone()
                        self.roots[reference.number]=(base,view_spec(value))
                        if self.key(value) in self.parameter_keys:self.root_parameters.add(reference.number)
                self.owners.setdefault(identity,reference)
                self.owner_specs.setdefault(identity,view_spec(value))
                return reference
            with torch._C._DisableTorchDispatch():return tree_map(tensor,tree)

        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            if self.opaque:return func(*args,**kwargs)
            if self.rng_cpu is None:
                with torch._C._DisableTorchDispatch():
                    self.rng_cpu=torch.get_rng_state()
                    if torch.cuda.is_available():self.rng_cuda=torch.cuda.get_rng_state()
            inputs=self.encode((args,kwargs))
            result=func(*args,**kwargs)
            outputs=self.encode(result,produced=True)
            step=dict(kind='aten',op=str(func),inputs=inputs,output=outputs,stage=self.stage)
            if str(func)=='aten.slice.Tensor':step['input_shape']=tuple(args[0].shape)
            self.program.append(step)
            return result

        def record(self,kind,inputs,outputs,**info):
            # Native kernels cannot be treated as recorded floating constants.
            # Their inputs and geometry form explicit replay operations.
            encoded=self.encode(inputs)
            super().record(kind,inputs,outputs,**info)
            event=self.events[-1]
            self.program.append(dict(kind='native',inputs=encoded,
                                     output=self.encode(outputs,produced=True),event=event))

        def attach(self):
            super().attach()
            import spconv.pytorch as spconv
            try:import MinkowskiEngine as ME
            except ImportError:ME=None
            for name,module in self.model.named_modules():
                if isinstance(module,spconv.SubMConv3d):
                    def before(mod,args,name=name):
                        value=args[0]
                        with torch._C._DisableTorchDispatch():
                            carrier=value.replace_feature(torch.empty((0,value.features.shape[1]),
                                dtype=value.features.dtype,device=value.features.device))
                        self.carriers[name]=carrier
                    self.handles.append(module.register_forward_pre_hook(before))
                elif ME and isinstance(module,(ME.MinkowskiConvolution,ME.MinkowskiConvolutionTranspose)):
                    def before(mod,args,name=name):
                        value=args[0]
                        self.carriers[name]=(value.coordinate_map_key,value.coordinate_manager)
                    self.handles.append(module.register_forward_pre_hook(before))
    return Recorder


class Replay:
    def __init__(self,trace,output,k_tile=None,sparse_column_config=None,dense_column_config=None,
                 attention_config=None,observations=None):
        self.trace=trace;self.output=output
        self.k_tile=k_tile
        self.sparse_column_config=sparse_column_config
        self.dense_column_config=dense_column_config
        self.attention_config=attention_config
        self.last={}
        for i,step in enumerate(trace.program):
            tree=step.get('inputs',step.get('base'))
            for value in tree_flatten(tree)[0]:
                if isinstance(value,Ref):self.last[value.number]=i
        self.last[output.number]=len(trace.program)
        self.observations=observations or {}
        for ref in self.observations.values():self.last[ref.number]=len(trace.program)
        self.plans={}

    def sdpa(self,op,args,kwargs):
        """Finite query windows; FP32 scores/AV and source-dtype probabilities.

        The deterministic native call supplies backend-specific tuple metadata
        only. Both floating feature outputs are replaced by explicit arithmetic
        from Q/K/V. Native replay execution time is not a hardware measurement.
        """
        c=self.attention_config
        q,k,v=args[:3];name=str(op)
        if (q.shape[-2:]!=k.shape[-2:] or q.shape[-2:]!=v.shape[-2:]):
            raise NotImplementedError('Attention lowering requires equal Q/K/V sequences and widths')
        public=name=='aten.scaled_dot_product_attention.default'
        defaults=dict(attn_mask=None,dropout_p=0.,is_causal=False,return_debug_mask=False,enable_gqa=False)
        positional=('attn_mask','dropout_p','is_causal') if public else ('dropout_p','is_causal','return_debug_mask')
        defaults.update(zip(positional,args[3:]));defaults.update(kwargs)
        if (defaults['attn_mask'] is not None or defaults['dropout_p'] or defaults['is_causal']
            or defaults['return_debug_mask'] or defaults['enable_gqa']):
            raise NotImplementedError('Attention lowering supports deterministic unmasked inference only')
        native=op(*args,**kwargs)
        native_output=native if isinstance(native,torch.Tensor) else native[0]
        if native_output.storage_offset():raise NotImplementedError('Nonzero backend output storage offset')
        scale=defaults.get('scale')
        if scale is None:scale=q.shape[-1]**-.5
        query_tiles=(1 if c.get('consumer_attention_pipeline',False) or
                     c.get('consumer_attention_pipeline_active',False) else c.get('attention_query_tiles',1))
        query_rows=c['array_rows']*query_tiles
        output=torch.empty_strided(native_output.shape,native_output.stride(),device=q.device,dtype=q.dtype)
        if isinstance(native,torch.Tensor):lse=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
        else:
            if native[1].shape!=q.shape[:-1] or native[1].storage_offset():
                raise NotImplementedError('Unsupported backend logsumexp shape or storage offset')
            lse=torch.empty_strided(native[1].shape,native[1].stride(),device=q.device,dtype=torch.float32)
        for begin in range(0,q.shape[-2],query_rows):
            high=min(q.shape[-2],begin+query_rows)
            score=(q[...,begin:high,:].float()@k.float().transpose(-1,-2))*scale
            maximum=score.max(-1,keepdim=True).values
            exponent=(score-maximum).exp();total=exponent.sum(-1,keepdim=True)
            probability=(exponent/total).to(q.dtype)
            output[...,begin:high,:]=(probability.float()@v.float()).to(q.dtype)
            lse[...,begin:high]=(maximum+total.log()).squeeze(-1)
        self.lowered_attention_operations+=1
        if isinstance(native,torch.Tensor):return output
        result=list(native);result[0]=output
        if result[1].shape!=lse.shape:
            raise NotImplementedError('Unsupported backend logsumexp shape')
        result[1]=lse
        return tuple(result)

    def linear(self,source,weight,bias=None):
        """FP32 K contributions with the simulated final output cast/bias.

        Validation only: no timing or claim of extra hardware buffers. Each
        product uses the actual FP16/FP32 operand values, K order, N boundary
        and one final output cast, matching retained PE accumulation.
        """
        c=self.dense_column_config
        if c is None:return F.linear(source,weight,bias)
        columns=weight.shape[0];inner=source.shape[-1]
        stripe=(c.get('dense_output_columns',0) or columns)
        if c.get('dense_columns_oversized_only',False) and weight.numel()*weight.element_size()<=c['weight_sram_bytes']:
            stripe=columns
        depth=self.k_tile or c['k_tile']
        depth//=c.get('consumer_reduction_slices',1)
        if (c.get('consumer_packed_operands',False) and
                c.get('reuse_operand_tiles',False) and
                not c.get('stream_operand_slices',False) and
                source.dtype==weight.dtype==torch.float16):
            depth*=2
        if depth<=0 or stripe<=0:raise ValueError('Positive dense K and N tiles required')
        lhs=source.reshape(-1,inner)
        output=torch.empty((len(lhs),columns),dtype=source.dtype,device=source.device)
        for n in range(0,columns,stripe):
            end=min(columns,n+stripe)
            total=torch.zeros((len(lhs),end-n),dtype=torch.float32,device=source.device)
            for k in range(0,inner,depth):
                high=min(inner,k+depth)
                total.add_(lhs[:,k:high].float()@weight[n:end,k:high].float().T)
            if bias is not None:total.add_(bias[n:end].float())
            output[:,n:end]=total.to(source.dtype)
        self.lowered_dense_operations+=1
        return output.reshape(*source.shape[:-1],columns)

    @staticmethod
    def mapped_affine(event,values):
        """Reexecute the fused diagnostic merge from actual bank operands.

        The recorded maps are tensor inputs, including runtime compaction.
        No captured floating output or diagnostic row list supplies a value.
        Bounded windows only limit validation memory; they add no hardware.
        """
        destination=values[:2] if event.get('diagnostic_affine_scatter',False) else None
        if destination is not None:values=values[2:]
        widths=event.get('diagnostic_centered_partition_widths',[])
        centered=bool(widths);fields=4 if centered else 2
        response=values[-3:] if event.get('diagnostic_affine_response',False) else None
        if response is not None:values=values[:-3]
        banks,remainder=divmod(len(values)-1,fields+1)
        if remainder or not 1<=banks<=5 or centered and len(widths)!=banks:
            raise ValueError('Invalid mapped affine operand count')
        bias=values[fields*banks].float()
        maps=values[fields*banks+1:]
        # Compaction can change under lowered arithmetic. The map tensors
        # supply the actual current extent; captured output sizes do not.
        count=maps[0].numel()
        if any(mapping.numel()!=count for mapping in maps):
            raise ValueError('Mapped affine input rows differ from output')
        dtype=getattr(torch,event['outputs'][0]['dtype'].split('.')[-1])
        output=torch.empty((count,bias.shape[1]),device=bias.device,dtype=dtype)
        for begin in range(0,count,8192):
            end=min(count,begin+8192)
            result=torch.zeros((end-begin,bias.shape[1]),device=bias.device,dtype=torch.float32)
            square=torch.zeros((end-begin,1),device=bias.device,dtype=torch.float32)
            mean=torch.zeros_like(square);local=[]
            for bank,mapping in enumerate(maps):
                ids=mapping.flatten()[begin:end].long();offset=fields*bank
                result.add_(values[offset][ids].float())
                square.add_(values[offset+1][ids].float())
                if centered:
                    mu=values[offset+2][ids].float()
                    mean.add_(mu*widths[bank]);local.append(mu)
            if centered:
                mean.div_(sum(widths))
                for bank,mu in enumerate(local):
                    delta=mu-mean
                    result.add_(delta*values[fields*bank+3].float())
                    square.add_(widths[bank]*delta.square())
            scale=square.sqrt().clamp_min(1e-12)
            value=(result/scale+bias if event['diagnostic_partitioned_output']
                   else result+scale*bias)
            if event.get('diagnostic_partition_normalized',False):value=F.normalize(value,dim=1)
            if response is not None:
                final,prefix,owners=response;ids=owners.flatten()[begin:end].long()
                value=F.normalize((value+final[ids])-prefix[ids],dim=1)
            output[begin:end]=value.to(dtype)
        if destination is not None:
            target,ids=destination
            target[ids.flatten().long()]=output
            return [target]
        return [output]

    def native(self,step,values,lowered):
        event=step['event'];kind=event['kind']
        source=values[0]
        if event.get('name')=='chorus.mapped_affine_merge':
            return self.mapped_affine(event,values)
        if event.get('name')=='chorus.response_gate':
            slots=event['response_owner_slots'];axes=event['response_gate_axes']
            previous,current,corrected,pilot,at,peers,positions=values[:7]
            winner=values[7].flatten()
            edges=[e.flatten() for e in values[7+slots:7+slots+axes]]
            at=at.flatten().long()
            if event.get('response_gate_folded_scores',False):
                corrected=(current+corrected[at])-values[7+slots+axes][at]
            old=F.normalize(previous,dim=1);now=F.normalize(current,dim=1)
            new=F.normalize(corrected,dim=1);top=new.topk(2,dim=1).values
            predicted=corrected.argmax(1)
            agreement=torch.zeros(len(current),device=current.device,dtype=torch.long)
            for slot in range(slots):
                agreement+=((peers[:,slot]>=0)&(winner[positions[:,slot].long()]==predicted)).long()
            features=[(now-old).abs().sum(1),top[:,0]-top[:,1],(now-pilot[at]).abs().sum(1)]
            if event['response_gate_magnitude']:features.append(values[-1].flatten()[at])
            cell=agreement
            for edge,feature in zip(edges,features):cell=cell*(len(edge)+1)+torch.searchsorted(edge,feature.contiguous())
            return [cell[:,None]]
        if event.get('name')=='chorus.response_owner':
            slots=event['response_owner_slots'];current=values[0];pilot=values[1]
            peers,positions=values[-2:]
            best=current.new_full((len(current),),torch.inf)
            selected=torch.full((len(current),),-1,device=current.device,dtype=torch.long)
            owner=torch.zeros_like(selected)
            for slot in range(slots):
                producer=positions[:,slot].long()
                distance=(current-pilot[producer]).abs().sum(1)
                take=(peers[:,slot]>=0)&(distance<best)
                best=torch.where(take,distance,best)
                selected=torch.where(take,peers[:,slot],selected)
                owner=torch.where(take,producer,owner)
            return [torch.stack((selected,owner),dim=1)]
        if event.get('name')=='chorus.response_refine':
            language,final,prefix,mapping=values
            owners=mapping.flatten().long()
            return [F.normalize((language+final[owners])-prefix[owners],dim=1)]
        if event.get('op')=='gadra.unique_index':
            # Recompute IDs/inverse from the runtime map tensor. The event's
            # captured diagnostic_unique_native list is timing topology only.
            used,inverse=torch.unique(source.flatten(),sorted=True,return_inverse=True)
            if (len(used) and (int(used[0])<0 or
                    int(used[-1])>=event['diagnostic_unique_rows'])):
                raise ValueError('Runtime native index outside source domain')
            return [torch.cat((used,inverse)).to(torch.int32)[None,:]]
        if kind=='progressive_difference':
            current,pilot,mapping,ids=values;begin=event['begin']
            mapping=mapping.flatten();ids=ids.flatten()
            return [(current.float()-pilot[mapping[ids],begin:begin+current.shape[1]].float()).half()]
        if kind=='progressive_update':
            output,contribution,ids=values
            ids=ids.flatten()
            output[ids]+=contribution.float()
            observed=output[ids] if event.get('materialize',True) else None
            if observed is not None and event.get('half_observation',True):observed=observed.half().float()
            return [output,observed]
        if kind=='progressive_observe':
            from .progressive import observe_reference
            constants={k:event[k] for k in ('window','bins','depth','depths','tolerance','force_full')}
            operands=list(values)
            for j in (1,2,8,9):operands[j]=operands[j].flatten()
            score,keep=observe_reference(*operands,**constants)
            return [score,keep[:,None],values[3],values[4]]
        if kind=='ancestor_merge':
            channels=event['channels'];width=sum(channels)
            maps=[x[:,0].long() for x in values[5:10]]
            means=[x[:,0][ids] for x,ids in zip(values[10:15],maps)]
            variances=[x[:,0][ids] for x,ids in zip(values[15:20],maps)]
            mean=sum(c*mu for c,mu in zip(channels,means))/width
            variance=sum(c*(var+(mu-mean).square()) for c,mu,var in zip(channels,means,variances))/width
            lo,hi=([0,len(values[20])] if event.get('sliced_inputs') else event.get('columns',[0,len(values[20])]))
            parts=[p[:,lo:hi] for p in values[:5]]
            if event['inline_fine']:
                # The fused kernel projects only this output window's fine
                # rows. Projecting the entire bank here invents unused work
                # and a scene-sized validation temporary for every window.
                fine=values[0][maps[0]]
                projected=(self.linear(fine,values[22][lo:hi]) if lowered else
                           F.linear(fine,values[22][lo:hi])).float()
                for partial,ids in zip(parts[1:],maps[1:]):projected=projected+partial[ids].float()
            else:projected=sum(p[ids].float() for p,ids in zip(parts,maps))
            pre=(projected-mean[:,None]*values[20][lo:hi])*torch.rsqrt(variance[:,None]+event['eps'])+values[21][lo:hi]
            return [F.gelu(pre.half())]
        if kind=='geometry' and event.get('name')=='chorus.serialization':
            from chorus_serialization import fused_encode
            return [fused_encode(*values,depth=event['depth'],order=event['order'])]
        if kind=='coordinate_map':
            from measure_embodiedsplat_native import map_rows
            target=self.trace.maps[event['map_key']]['target_coordinates'].numpy()
            coordinate=source.cpu().numpy()
            if len(target)!=len(coordinate):
                raise NotImplementedError('Replay of duplicate-coordinate quantization needs its reduction policy')
            mapping=map_rows(target,coordinate,[1,1,1])
            if len(np.unique(mapping))!=len(mapping):raise ValueError('Coordinate insertion is not bijective')
            unique=torch.as_tensor(mapping,device=source.device,dtype=torch.int64)
            inverse=unique.argsort()
            return [unique.to(getattr(torch,event['outputs'][0]['dtype'].split('.')[-1])),
                    inverse.to(getattr(torch,event['outputs'][1]['dtype'].split('.')[-1]))]
        if kind=='coordinate_decomposition':
            from measure_embodiedsplat_native import map_rows
            coordinates=source.cpu().numpy()
            targets=self.trace.maps[event['map_key']]['target_coordinates']
            return [torch.as_tensor(map_rows(target.numpy(),coordinates,[1,1,1]),
                device=source.device,dtype=getattr(torch,spec['dtype'].split('.')[-1]))
                for target,spec in zip(targets,event['outputs'])]
        if kind=='linear':
            if event.get('indexed_source',False):
                state,weight=values[:2];ids=values[-1]
                bias=values[2] if len(values)==4 else None
                values=[state[ids.flatten().long()],weight,bias]
            return [(self.linear(*values) if lowered else F.linear(*values))]
        if event.get('name')=='Minkowski.features_at_coordinates':
            key=event['name'],id(event)
            if key not in self.plans:self.plans[key]=event['_index_values'][0].to(source.device)
            return [source.index_select(0,self.plans[key])]
        if kind=='ME' and not lowered:
            import MinkowskiEngine as ME
            coordinate_key,manager=self.trace.carriers[event['name']]
            sparse=ME.SparseTensor(source,coordinate_map_key=coordinate_key,coordinate_manager=manager)
            return [self.trace.model.get_submodule(event['name'])(sparse).F]
        if kind=='spconv' and not lowered:
            carrier=self.trace.carriers[event['name']]
            return [self.trace.model.get_submodule(event['name'])(carrier.replace_feature(source)).features]
        if kind not in ('spconv','ME'):
            raise NotImplementedError(f"Native replay missing kind={kind}, op={event.get('op')}, name={event.get('name')}")
        name=event['map_key'];rule=self.trace.maps[name]
        if name not in self.plans:
            if kind=='spconv':
                pairs=[]
                for row in rule['mapping']:
                    targets=(row>=0).nonzero().flatten()
                    pairs.append((row[targets].long().to(source.device),targets.to(source.device)))
            else:pairs=[(p[0].long().to(source.device),p[1].long().to(source.device)) for p in rule['offsets']]
            self.plans[name]=pairs
        width=event['outputs'][0]['shape'][-1]
        output_rows=event['outputs'][0]['shape'][0]
        result=torch.zeros((output_rows,width),dtype=torch.float32,device=source.device)
        weight=values[1]
        if kind=='spconv':weight=weight.reshape(width,-1,source.shape[-1]).permute(1,2,0)
        stripe=width
        if self.sparse_column_config:
            from .feeder import sparse_stripe,use_sparse_columns
            if use_sparse_columns(self.sparse_column_config,event,output_rows):
                stripe=sparse_stripe(self.sparse_column_config,event)
        for offset,(inputs,targets) in enumerate(self.plans[name]):
            if not len(inputs):continue
            # FP32 products/accumulation are the explicit numeric contract of
            # this lowering. It must pass end-to-end against native kernels.
            selected=source.index_select(0,inputs).float()
            for begin in range(0,source.shape[-1],self.k_tile or source.shape[-1]):
                end=min(source.shape[-1],begin+(self.k_tile or source.shape[-1]))
                for column in range(0,width,stripe):
                    high=min(column+stripe,width)
                    contribution=selected[:,begin:end]@weight[offset,begin:end,column:high].float()
                    result[:,column:high].index_add_(0,targets,contribution)
        if len(values)>2 and values[2] is not None:result+=values[2].float()
        result=result.to(source.dtype)
        native=self.native(step,values,False)[0]
        difference=(result.float()-native.float())
        self.local_errors.append((event['name'],torch.stack((difference.abs().max(),
            difference.square().mean().sqrt(),native.float().square().mean().sqrt()))))
        return [result]

    @torch.inference_mode()
    def run(self,lowered=False,observer=None):
        # The models reset their seed in Python before the first ATen call.
        # Serialization permutations are real operations and must replay from
        # that initial RNG state, not from the post-capture state.
        if self.trace.rng_cpu is not None:torch.set_rng_state(self.trace.rng_cpu)
        if self.trace.rng_cuda is not None:torch.cuda.set_rng_state(self.trace.rng_cuda)
        self.local_errors=[]
        self.lowered_dense_operations=0
        self.lowered_attention_operations=0
        self.native_attention_operations=0
        env={key:storage_view(base.clone(),spec) for key,(base,spec) in self.trace.roots.items()}
        if observer is not None:observer.roots(env,getattr(self.trace,'root_parameters',set()))
        unique_bounds={}
        recent_runtime=[]
        def resolve(tree):return tree_map(lambda x:env[x.number] if isinstance(x,Ref) else x,tree)
        for i,step in enumerate(self.trace.program):
            if step['kind']=='rng_reset':
                torch.manual_seed(step['seed'])
                continue
            if step['kind']=='alias':
                base=resolve(step['base']);spec=step['spec'];prior=step.get('base_spec')
                singleton=(len(spec['shape'])==2 and 1 in spec['shape'] and
                           spec['stride'][1]==1 and
                           (spec['shape'][0]==1 or spec['stride'][0]==1))
                if (prior is not None and singleton and spec['dtype']==base.dtype
                    and base.numel()!=math.prod(prior['shape'])):
                    offset=spec['offset']-prior['offset'];old_size=math.prod(prior['shape'])
                    if offset+math.prod(spec['shape'])==old_size:
                        # Opaque kernels receive a final-tail map view. The
                        # alias follows the runtime producer's actual length.
                        boundary=unique_bounds.get(step['base'].number)
                        if boundary is not None and offset==boundary[0]:offset=boundary[1]
                        size=base.numel()-offset
                        shape,stride=(((size,1),(1,1)) if spec['shape'][1]==1
                                      else ((1,size),(size,1)))
                        spec=dict(spec,shape=shape,stride=stride,offset=base.storage_offset()+offset)
                result=storage_view(base,spec)
            elif step['kind']=='aten':
                namespace,name,overload=step['op'].split('.')
                op=getattr(getattr(getattr(torch.ops,namespace),name),overload)
                args,kwargs=resolve(step['inputs'])
                try:
                    if step['op']=='aten.slice.Tensor' and 'input_shape' in step:
                        args=list(args);dim=args[1] if len(args)>1 else 0
                        old_extent=step['input_shape'][dim];new_extent=args[0].shape[dim]
                        # A full final window includes the new runtime tail.
                        # Interior window boundaries keep their fixed size.
                        if len(args)>3 and args[3]==old_extent:args[3]=new_extent
                        source_ref=step['inputs'][0][0]
                        boundary=unique_bounds.get(source_ref.number) if isinstance(source_ref,Ref) else None
                        if boundary is not None and dim==args[0].ndim-1:
                            old_count,new_count=boundary
                            if len(args)>2 and args[2]==old_count:args[2]=new_count
                            if len(args)>3 and args[3]==old_count:args[3]=new_count
                    if (lowered and self.dense_column_config is not None and step['op']=='aten.sort.default'
                        and not args[0].is_floating_point() and args[0].ndim==1):
                        from .integer_stream import needed
                        event=dict(control_primitive='sort',inputs=[dict(shape=list(args[0].shape))])
                        if needed(event,self.dense_column_config):
                            dim=args[1] if len(args)>1 else kwargs.get('dim',-1)
                            descending=args[2] if len(args)>2 else kwargs.get('descending',False)
                            result=torch.sort(args[0],dim=dim,descending=descending,stable=True)
                        else:result=op(*args,**kwargs)
                    elif (lowered and self.attention_config is not None and step['op'] in (
                        'aten.scaled_dot_product_attention.default','aten._scaled_dot_product_flash_attention.default',
                        'aten._scaled_dot_product_flash_attention_for_cpu.default')):
                        if (self.attention_config.get('attention_register_softmax',False)
                            and args[0].shape[-2]<=self.attention_config['vector_lanes']):
                            result=self.sdpa(op,args,kwargs)
                        else:
                            self.native_attention_operations+=1;result=op(*args,**kwargs)
                    elif lowered and self.dense_column_config is not None and step['op']=='aten.linear.default':
                        result=self.linear(*args,**kwargs)
                    elif lowered and self.dense_column_config is not None and step['op']=='aten.mm.default':
                        result=self.linear(args[0],args[1].T)
                    elif (lowered and self.dense_column_config is not None and step['op']=='aten.addmm.default'
                          and kwargs.get('alpha',1)==1 and kwargs.get('beta',1)==1 and args[0].ndim==1):
                        result=self.linear(args[1],args[2].T,args[0])
                    else:result=op(*args,**kwargs)
                except Exception as error:
                    shapes=[tuple(value.shape) for value in tree_flatten((args,kwargs))[0] if isinstance(value,torch.Tensor)]
                    recent=[(j,s.get('op',s.get('event',{}).get('name')),s.get('input_shape'),
                             s.get('spec'),s.get('base_spec'))
                            for j,s in enumerate(self.trace.program[max(0,i-14):i],max(0,i-14))]
                    raise RuntimeError(f"Replay step {i}, {step['op']}, tensor shapes {shapes}; recent operations {recent}; runtime {recent_runtime}") from error
            else:
                args=resolve(step['inputs']);result=self.native(step,args,lowered)
            if observer is not None and step['kind']!='alias':
                observer.step(step,args,kwargs if step['kind']=='aten' else {},result)
            refs=tree_flatten(step['output'])[0];actual=tree_flatten(result)[0]
            recent_runtime.append((i,[tuple(v.shape) for v in actual if isinstance(v,torch.Tensor)]))
            recent_runtime=recent_runtime[-14:]
            if len(refs)!=len(actual):raise AssertionError(('Output tree mismatch',i))
            if step.get('event',{}).get('op')=='gadra.unique_index':
                input_count=resolve(step['inputs'])[0].numel()
                old_count=(math.prod(step['event']['outputs'][0]['shape'])-
                           math.prod(step['event']['inputs'][0]['shape']))
                new_count=actual[0].numel()-input_count
                unique_bounds[refs[0].number]=(old_count,new_count)
            elif step.get('op')=='aten.select.int':
                source_ref=step['inputs'][0][0]
                if isinstance(source_ref,Ref) and source_ref.number in unique_bounds:
                    unique_bounds[refs[0].number]=unique_bounds[source_ref.number]
            for ref,value in zip(refs,actual):
                if isinstance(ref,Ref):env[ref.number]=value
            used=tree_flatten(step.get('inputs',step.get('base')))[0]
            for ref in used:
                if isinstance(ref,Ref) and self.last[ref.number]==i:env.pop(ref.number,None)
            for ref in refs:
                if isinstance(ref,Ref) and ref.number not in self.last:env.pop(ref.number,None)
        self.observed={name:env[ref.number] for name,ref in self.observations.items()}
        return env[self.output.number]


def metrics(actual,reference,text,tolerance=.01):
    a=F.normalize(actual.float(),dim=-1);b=F.normalize(reference.float(),dim=-1)
    score=a@text.T;target=b@text.T;chosen=score.argmax(1)
    regret=target.max(1).values-target.gather(1,chosen[:,None]).squeeze(1)
    agreement=int((chosen==target.argmax(1)).sum())/len(chosen)
    within=int((regret<=tolerance).sum())/len(chosen)
    return dict(max_abs_error=float((actual.float()-reference.float()).abs().max()),
        rmse=float((actual.float()-reference.float()).square().mean().sqrt()),
        mean_cosine=float((a*b).sum(1).mean()),class_agreement=agreement,
        within_tolerance=within,passes_thresholds=agreement>=.99 and within>=.99)


def bounded_metrics(actual,reference,text,tolerance=.01,*,row_window=8192):
    """Evaluate complete outputs in bounded device windows, outside timing.

    Outputs may reside on CPU to avoid retaining several full scene-sized
    CUDA language matrices. Every row is evaluated; only counts and FP64
    reduction totals survive a window. No future target enters inference.
    """
    if not isinstance(row_window,int) or isinstance(row_window,bool) or row_window<1:
        raise ValueError('Quality row window must be a positive integer')
    if actual.shape!=reference.shape or actual.ndim!=2 or not len(actual):
        raise ValueError('Quality requires nonempty matching language matrices')
    total=torch.zeros(5,device=text.device,dtype=torch.float64)
    maximum=torch.zeros((),device=text.device)
    for begin in range(0,len(actual),row_window):
        x=actual[begin:begin+row_window].to(device=text.device,dtype=torch.float32)
        y=reference[begin:begin+row_window].to(device=text.device,dtype=torch.float32)
        a=F.normalize(x,dim=-1);b=F.normalize(y,dim=-1)
        score=a@text.T;target=b@text.T;chosen=score.argmax(1)
        regret=target.max(1).values-target.gather(1,chosen[:,None]).squeeze(1)
        difference=x-y
        maximum=torch.maximum(maximum,difference.abs().max())
        total+=torch.stack((difference.square().sum(dtype=torch.float64),
            (a*b).sum(1).sum(dtype=torch.float64),
            (chosen==target.argmax(1)).sum().double(),(regret<=tolerance).sum().double(),
            regret.sum(dtype=torch.float64)))
    maximum=float(maximum)
    square,cosine,correct,within,regret=total.cpu().tolist()
    n=len(actual);agreement=int(correct)/n;inside=int(within)/n
    return dict(max_abs_error=maximum,rmse=(square/actual.numel())**.5,
        mean_cosine=cosine/n,class_agreement=agreement,within_tolerance=inside,
        mean_final_score_regret=regret/n,correct_rows=int(correct),within_rows=int(within),
        rows=n,passes_thresholds=agreement>=.99 and inside>=.99)


@torch.inference_mode()
def validate(family,dataset,index,destination,k_tile=None,ancestor_affine=False,progressive_variant=None,query_proxy=False,parent_window=None,mixed_submanifold=False,sparse_column_config=None,mixed_extended=False,mixed_wide_kernel=False,
             *,ancestor_stripe_columns=None,ancestor_full=False,dense_column_config=None,attention_config=None):
    if ancestor_full:
        if ancestor_stripe_columns is not None:raise ValueError('Full and striped ancestor candidates are exclusive')
        ancestor_affine=True
    if ancestor_stripe_columns is not None:
        if ancestor_stripe_columns<1:raise ValueError('Ancestor stripe width must be positive')
        ancestor_affine=True
    if mixed_wide_kernel and not mixed_extended:
        raise ValueError('Wide-kernel precision requires the extended mixed candidate')
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model,call,context,_=load_case(family,dataset,index)
    allowed={int(v.untyped_storage()._cdata) for v in tree_flatten(
        model._gadra_inputs if family!='EmbodiedSplat' else model._gadra_native[2])[0] if isinstance(v,torch.Tensor)}
    official=None
    if family!='EmbodiedSplat':
        if family=='SceneSplat':
            from scenesplat_runtime import official_modules
            from profile_scenesplat import texts
        else:from chorus_runtime import official_modules,texts
        official=official_modules();text=texts(dataset).cuda()
    else:text=torch.load(WORKLOADS/'results/embodiedsplat_native_measurement/texts.pt',weights_only=False)[dataset]['features'].cuda()
    try:
        with ExitStack() as stack:
            stack.enter_context(context)
            reference=call().clone()
            if mixed_submanifold or mixed_extended:
                if family!='EmbodiedSplat':raise ValueError('Mixed submanifold candidate requires EmbodiedSplat')
                from .precision import submanifold_fp16
                stack.enter_context(submanifold_fp16(model,extended=mixed_extended,wide_kernel=mixed_wide_kernel))
            if ancestor_affine:
                if family!='Chorus':raise ValueError('Ancestor affine applies only to Chorus')
                from .factorized import ancestor_projector
                stack.enter_context(ancestor_projector(model,projection_dtype=torch.float16,use_fused=True,
                    inline_fine=ancestor_full or ancestor_stripe_columns is not None,stripe_columns=ancestor_stripe_columns))
            if progressive_variant:
                if family!='Chorus' or ancestor_affine:raise ValueError('Progressive candidate requires Chorus without the other projector context')
                from .progressive import progressive_projector,calibrated_lut,observation_root
                root=observation_root('results/raw/progressive_actual',True,query_proxy)
                lut=calibrated_lut(dataset,progressive_variant,root)
                stack.enter_context(progressive_projector(model,text,lut,progressive_variant=='A_ancestor',
                                                         fused_software=True,query_proxy=query_proxy,parent_window=parent_window))
            trace=recorder_type(family)(model,official,allowed)
            trace.attach()
            try:
                with trace:captured=call()
                output=trace.encode(captured)
            finally:trace.close()
            capture_error=float((reference.float()-captured.float()).abs().max())
            del captured
            trace.storage_handles.clear();trace.held_storages.clear();gc.collect();torch.cuda.empty_cache()
            replay=Replay(trace,output,k_tile=k_tile,sparse_column_config=sparse_column_config,
                          dense_column_config=dense_column_config,attention_config=attention_config)
            report=dict(model=family,dataset=dataset,index=index,steps=len(trace.program),
                root_tensors=len(trace.roots),capture_max_abs_error=capture_error,
                sparse_accumulation_k_tile=k_tile,
                sparse_column_config=sparse_column_config,
                dense_column_config=dense_column_config,
                attention_config=attention_config,
                ancestor_affine=ancestor_affine,
                ancestor_stripe_columns=ancestor_stripe_columns,
                ancestor_full=ancestor_full,
                mixed_submanifold=mixed_submanifold,
                mixed_extended=mixed_extended,
                mixed_wide_kernel=mixed_wide_kernel,
                progressive_variant=progressive_variant,
                query_proxy=query_proxy,
                parent_window=parent_window,
                scope='Full 3D feature replay from initial tensors; no captured intermediate feature roots')
            for lowered in (False,True):
                result=replay.run(lowered)
                report['lowered' if lowered else 'native_replay']=metrics(result,reference,text)
                if lowered:
                    report['lowered_dense_operations']=replay.lowered_dense_operations
                    report['lowered_attention_operations']=replay.lowered_attention_operations
                    report['native_attention_operations']=replay.native_attention_operations
                print(family,dataset,'lowered' if lowered else 'native',report['lowered' if lowered else 'native_replay'],flush=True)
                del result
            if replay.local_errors:
                values=torch.stack([value for _,value in replay.local_errors]).cpu().tolist()
                report['local_sparse_errors']=[dict(name=name,max_abs_error=error[0],
                    rmse=error[1],native_rms=error[2]) for (name,_),error in zip(replay.local_errors,values)]
            path=Path(destination);path.mkdir(parents=True,exist_ok=True)
            (path/f'{family}_{dataset}_s{index}_numerical.json').write_text(json.dumps(report,indent=2)+'\n')
            return report
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
