"""Real frozen-model trace and checkpoint observations; no checksum workflow."""
import fcntl
import json
import math
from pathlib import Path
import re
import time
from contextlib import ExitStack,contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from .workloads import WORKLOADS, load_case


class WeakStorageOwners(dict):
    """Keep StorageImpl identities unique without retaining their CUDA data.

    Execution-only capture and opt-in functional recording use this. Functional
    recording separately clones roots and records all produced values/aliases.
    PyTorch's weak storage reference outlives the allocation, preventing reuse
    of the _cdata identity while allowing the native allocator to free data.
    """
    def __setitem__(self, identity, storage):
        from torch.multiprocessing.reductions import StorageWeakRef
        super().__setitem__(identity, StorageWeakRef(storage))


def normalization_metadata(event,args,kwargs):
    if event.get('op')!='aten.batch_norm.default':return
    event['batch_norm_training']=bool(args[5] if len(args)>5 else kwargs['training'])
    event['has_running_statistics']=(args[3] is not None and args[4] is not None)


def segment_metadata(event,args,kwargs):
    if event.get('op')!='aten.segment_reduce.default':return
    from .segments import REDUCTIONS
    source=args[0];reduction=args[1] if len(args)>1 else kwargs['reduce']
    offsets=kwargs.get('offsets')
    axis=kwargs.get('axis',0)
    if (source.ndim!=2 or axis not in (0,-2) or reduction not in REDUCTIONS
        or offsets is None or offsets.ndim!=1 or not offsets.is_cuda
        or kwargs.get('initial') is not None or kwargs.get('lengths') is not None
        or kwargs.get('indices') is not None):return
    # A row view preserves the original pointer storage/address/stride while
    # allowing exactly the selected start/end words to be charged at runtime.
    if len(event['inputs'])!=2:return
    pointer=event['inputs'][1]
    pointer.update(shape=[offsets.numel(),1],stride=[offsets.stride(0),1])
    event.update(segment_axis=0,segment_reduction=reduction)
    with torch._C._DisableTorchDispatch():
        event['_index_values']=[offsets.detach().cpu().clone()]


def trace_type(family):
    from capture_memory_flow import TensorFlow
    class Trace(TensorFlow):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            begin=len(self.events)
            result=super().__torch_dispatch__(func,types,args,kwargs)
            for event in self.events[begin:]:
                normalization_metadata(event,args,kwargs or {})
                segment_metadata(event,args,kwargs or {})
                from .reductions import capture as reduction_metadata
                reduction_metadata(event,args,kwargs or {})
                from .control_timing import capture as control_metadata
                control_metadata(event,args,kwargs or {})
            if len(self.events)>begin and str(func)=='aten.index.Tensor':
                source,indices=args[:2]
                if (source.ndim==2 and len(indices)==1 and indices[0] is not None and
                    indices[0].ndim==1 and indices[0].dtype in (torch.int32,torch.int64)):
                    ids=indices[0]
                    event=self.events[-1]
                    event['row_selector']=True
                    with torch._C._DisableTorchDispatch():
                        event['_index_values']=[ids.detach().cpu().clone()]
                        if not ids.is_cuda:
                            identity=self.key(ids)
                            if identity not in self.storages:
                                self.storage_handles[identity]=ids.untyped_storage()
                                self.storages[identity]=dict(id=len(self.storages)+1,
                                    address=int(ids.untyped_storage().data_ptr()),
                                    bytes=ids.untyped_storage().nbytes(),parameter=False)
                            storage=self.storages[identity]
                            # Unified-memory index input, represented as one
                            # integer per output row. Values remain captured
                            # from the real call, never inferred from features.
                            event['inputs'].append(dict(storage=storage['id'],version=self.versions[identity],
                                address=int(ids.data_ptr()),shape=[len(ids),1],stride=[ids.stride(0),1],
                                element_bytes=ids.element_size(),dtype=str(ids.dtype),role='index',
                                contiguous=ids.is_contiguous(),bytes=ids.numel()*ids.element_size(),host_index=True))
            if len(self.events)>begin and str(func)=='aten.index_put_.default':
                destination,indices,value=args[:3]
                accumulate=args[3] if len(args)>3 else (kwargs or {}).get('accumulate',False)
                if (not accumulate and destination.ndim==value.ndim==2 and len(indices)==1
                    and indices[0] is not None and indices[0].ndim==1
                    and value.shape==(len(indices[0]),destination.shape[1])):
                    # A row store overwrites only its addressed rows. It does
                    # not read or copy the full destination on every window.
                    with torch._C._DisableTorchDispatch():
                        ids=indices[0].detach().cpu().numpy()
                    if len(np.unique(ids))==len(ids) and (not len(ids) or ids.min()>=0):
                        event=self.events[-1];event['kind']='indexed_store'
                        index_spec=event['inputs'][1]
                        index_spec.update(shape=[len(ids),1],stride=[index_spec['stride'][0],1])
                        event['_index_values']=[ids]
            return result
        def attach(self):
            super().attach()
            if family=='Chorus':
                from chorus.models.utils import structure
                original=structure.encode
                def encode(grid_coord,batch=None,depth=16,order='z'):
                    self.opaque+=1
                    try:result=original(grid_coord,batch,depth,order)
                    finally:self.opaque-=1
                    self.record('geometry',[grid_coord,batch],[result],name='chorus.serialization',
                        op='garda.serialized_encode',depth=depth,order=order,
                        geometry_scalar_ops=len(grid_coord)*(11*depth+2+(22*depth+12 if order.startswith('hilbert') else 0)))
                    return result
                structure.encode=encode
                self.saved.append((structure,'encode',original))
                # Triton writes are opaque to ATen. Record the arithmetic
                # boundary and all real inputs rather than accepting the
                # allocated-but-unwritten output as a floating input root.
                from . import ancestor_triton
                original_merge=ancestor_triton.merge_gelu
                def ancestor_merge(affine,projections,statistics,begin,end,inline_fine=False,columns=None,selected=None,
                                   resolved_maps=None,projection_columns=None):
                    # Selected rows depend on actual online decisions. Keep
                    # these gathers in the replay program; they are not roots
                    # merely because the resulting tensors contain integers.
                    resolved=resolved_maps if resolved_maps is not None else [s[2][begin:end] if selected is None else s[2][selected] for s in statistics]
                    maps=[mapping[:,None] for mapping in resolved]
                    means=[s[0][:,None] for s in statistics]
                    variances=[s[1][:,None] for s in statistics]
                    lo,hi=(0,len(affine.constant)) if columns is None else columns
                    projected_first=0 if projection_columns is None else projection_columns[0]
                    parts=[p if inline_fine and j==0 else p[:,lo-projected_first:hi-projected_first] for j,p in enumerate(projections)]
                    correction=affine.mean_correction[lo:hi];constant=affine.constant[lo:hi]
                    fine_weight=affine.partials[0].weight[lo:hi]
                    self.opaque+=1
                    try:
                        result=original_merge(affine,projections,statistics,begin,end,inline_fine,columns,selected,resolved,
                                              projection_columns=projection_columns)
                    finally:self.opaque-=1
                    self.record('ancestor_merge',parts+maps+means+variances+
                        [correction,constant,fine_weight],[result],
                        name='chorus.ancestor_merge',channels=list(affine.channels),
                        eps=affine.eps,inline_fine=inline_fine,
                        columns=[lo,hi],sliced_inputs=True)
                    self.events[-1]['_index_values']=[m[:,0].detach() for m in maps]
                    return result
                ancestor_triton.merge_gelu=ancestor_merge
                self.saved.append((ancestor_triton,'merge_gelu',original_merge))
                from . import progressive,progressive_triton,diagnostics
                original_response_owner=progressive_triton.response_owner
                def response_owner(current,pilot,peers,positions):
                    self.opaque+=1
                    try:result=original_response_owner(current,pilot,peers,positions)
                    finally:self.opaque-=1
                    with torch._C._DisableTorchDispatch():
                        count,classes=current.shape;slots=peers.shape[1]
                        identity=list(range(count))
                        mapping=positions.detach().cpu().numpy()
                        # Every candidate query is really read, including
                        # the clamped invalid slot. Separate references keep
                        # their ordinary storage/port qualification intact.
                        self.record('aten',[current]+[pilot]*slots+[peers,positions],[result],
                            op='aten.sub.Tensor',name='chorus.response_owner',
                            response_owner_slots=slots,diagnostic_row_local=True,
                            diagnostic_control_rows=[identity]+[mapping[:,s].tolist() for s in range(slots)]+[identity,identity],
                            scalar_ops_per_row=slots*(3*classes+6),
                            scalar_dependency_cycles=slots*(4+int(np.ceil(np.log2(max(1,classes))))))
                    return result
                progressive_triton.response_owner=response_owner
                self.saved.append((progressive_triton,'response_owner',original_response_owner))
                original_response_gate=progressive_triton.response_gate
                def response_gate(previous,current,corrected,pilot,at,peers,positions,winner,edges,change=None,*,folded_scores=None):
                    self.opaque+=1
                    try:result=original_response_gate(previous,current,corrected,pilot,at,peers,positions,winner,edges,change,folded_scores=folded_scores)
                    finally:self.opaque-=1
                    with torch._C._DisableTorchDispatch():
                        count,classes=current.shape;slots=peers.shape[1]
                        identity=list(range(count));zero=[0]*count
                        mapping=at.detach().cpu().tolist();producers=positions.detach().cpu().numpy()
                        folded=folded_scores is not None
                        inputs=[previous,current,folded_scores[0] if folded else corrected,pilot,at[:,None],peers,positions]
                        maps=[identity,identity,mapping if folded else identity,mapping,identity,identity,identity]
                        inputs.extend([winner[:,None]]*slots)
                        maps.extend(producers[:,s].tolist() for s in range(slots))
                        inputs.extend(edge[None,:] for edge in edges);maps.extend([zero]*len(edges))
                        if folded:inputs.append(folded_scores[1]);maps.append(mapping)
                        if change is not None:inputs.append(change[:,None]);maps.append(mapping)
                        self.record('aten',inputs,[result],op='aten.add.Tensor',name='chorus.response_gate',
                            response_owner_slots=slots,response_gate_axes=len(edges),
                            response_gate_magnitude=change is not None,response_gate_folded_scores=folded,diagnostic_row_local=True,
                            diagnostic_control_rows=maps,
                            scalar_ops_per_row=(24+2*int(folded))*classes+24+4*slots+sum(len(e) for e in edges)+2*len(edges),
                            scalar_dependency_cycles=9*int(np.ceil(np.log2(max(1,classes))))+12+slots+sum(len(e) for e in edges)+2*int(folded))
                    return result
                progressive_triton.response_gate=response_gate
                self.saved.append((progressive_triton,'response_gate',original_response_gate))
                original_response_refine=progressive_triton.response_refine
                def response_refine(language,final,prefix,mapping):
                    self.opaque+=1
                    try:result=original_response_refine(language,final,prefix,mapping)
                    finally:self.opaque-=1
                    with torch._C._DisableTorchDispatch():
                        count,width=result.shape
                        actual=mapping.detach().cpu().tolist();identity=list(range(count))
                        self.record('aten',[language,final,prefix,mapping[:,None]],[result],
                            op='aten.add.Tensor',name='chorus.response_refine',
                            diagnostic_control_rows=[identity,actual,actual,identity],
                            diagnostic_row_local=True,
                            scalar_ops_per_row=5*width+2,
                            scalar_dependency_cycles=6+int(np.ceil(np.log2(max(1,width)))))
                    return result
                progressive_triton.response_refine=response_refine
                self.saved.append((progressive_triton,'response_refine',original_response_refine))
                original_native_index=diagnostics.native_index_payload
                def native_index_payload(mapping,source_rows):
                    self.opaque+=1
                    try:payload,count=original_native_index(mapping,source_rows)
                    finally:self.opaque-=1
                    with torch._C._DisableTorchDispatch():
                        self.record('aten',[mapping[None,:]],[payload],
                            op='garda.unique_index',name='chorus.native_index_payload',
                            diagnostic_unique_native=mapping.detach().cpu().tolist(),
                            diagnostic_unique_rows=source_rows)
                    return payload,count
                diagnostics.native_index_payload=native_index_payload
                self.saved.append((diagnostics,'native_index_payload',original_native_index))
                original_merge_affine=progressive_triton.merge_affine
                def progressive_merge_affine(terms,weight,divide_norm,centered,begin,end,normalize_output=False,*,response=None,destination=None):
                    self.opaque+=1
                    try:result=(original_merge_affine(terms,weight,divide_norm,centered,begin,end,normalize_output,response=response,destination=destination)
                                if response is not None else original_merge_affine(terms,weight,divide_norm,centered,begin,end,normalize_output))
                    finally:self.opaque-=1
                    with torch._C._DisableTorchDispatch():
                        inputs=[];maps=[];indices=[];count=end-begin
                        identity=list(range(count));zero=[0]*count
                        for partial,energy,mean,coefficient,width,mapping in terms:
                            index=mapping[begin:end,None]
                            actual=mapping[begin:end].detach().cpu().tolist()
                            inputs.extend((partial,energy));maps.extend((actual,actual))
                            if centered:
                                inputs.extend((mean,coefficient));maps.extend((actual,zero))
                            indices.append(index)
                        inputs.append(weight[-1:]);maps.append(zero)
                        inputs.extend(indices);maps.extend([identity for _ in indices])
                        banks=len(terms);classes=weight.shape[1]
                        # Same existing mapped vector microprogram as Program.partitioned_query.
                        from .diagnostics import affine_merge_cost
                        operations,dependency=affine_merge_cost(classes,banks,centered,divide_norm)
                        if normalize_output:
                            operations+=3*classes+2
                            dependency+=math.ceil(math.log2(classes))+3
                        if response is not None:
                            final,prefix,owner=response
                            actual=owner.detach().cpu().tolist()
                            inputs.extend((final,prefix,owner[:,None]));maps.extend((actual,actual,identity))
                            operations+=5*classes+2
                            dependency+=6+math.ceil(math.log2(classes))
                        if destination is not None:
                            output,output_ids=destination
                            # The first operand provides old-version byte
                            # inheritance only; its feature values are never
                            # loaded by the mapped response microprogram.
                            inputs=[output,output_ids[:,None]]+inputs
                            maps=[[],identity]+maps
                        self.record('indexed_store' if destination is not None else 'aten',inputs,[result],op='aten.mul.Tensor',
                            name='chorus.mapped_affine_merge',diagnostic_control_rows=maps,
                            scalar_ops_per_row=operations,
                            scalar_dependency_cycles=dependency,
                            diagnostic_centered_partition_widths=([t[4] for t in terms] if centered else []),
                            diagnostic_partitioned_output=divide_norm,
                            diagnostic_affine_response=response is not None,
                            diagnostic_affine_scatter=destination is not None,
                            diagnostic_partition_normalized=normalize_output)
                        if destination is not None:self.events[-1]['_index_values']=[output_ids.detach()]
                    return result
                progressive_triton.merge_affine=progressive_merge_affine
                self.saved.append((progressive_triton,'merge_affine',original_merge_affine))
                self.channel_checkpoints=[]
                original_checkpoint=progressive.checkpoint_program
                @contextmanager
                def checkpoint_program():
                    begin=len(self.events)
                    with original_checkpoint():yield
                    if len(self.events)>begin:
                        self.channel_checkpoints.append(dict(begin=begin,end=len(self.events)-1))
                progressive.checkpoint_program=checkpoint_program
                self.saved.append((progressive,'checkpoint_program',original_checkpoint))
                self.native_coverage_states=[]
                if hasattr(getattr(getattr(self.model,'projectors',None),'lang',None),'progressive'):
                    for name,module in self.model.named_modules():
                        matched=re.fullmatch(r'(?:backbone\.)?enc\.enc(\d+)\.block\d+',name)
                        if matched is None:continue
                        level=int(matched[1])
                        def native_state(mod,args,result,name=name,level=level):
                            self.native_coverage_states.append(dict(name=name,level=level,spec=self.spec(result.feat)))
                        self.handles.append(module.register_forward_hook(native_state))
                original_banks=progressive.context_banks
                def progressive_banks(point):
                    result=original_banks(point)
                    # Native maps already exist in this execution. Recording
                    # their row domains adds no output/diagnostic demands.
                    with torch._C._DisableTorchDispatch():
                        self.coverage_checkpoints=[dict(name=f'native_bank_{level}',
                            spec=self.spec(value),mapping=mapping.detach())
                            for level,(value,mapping) in enumerate(result[1])]
                        self.coverage_checkpoints.extend(dict(name=record['name'],spec=record['spec'],
                            mapping=result[1][record['level']][1].detach()) for record in self.native_coverage_states)
                    return result
                progressive.context_banks=progressive_banks
                self.saved.append((progressive,'context_banks',original_banks))
                def indexed_affine_capture(original,name):
                    def execute(state,weight,bias,ids):
                        id_view=ids[:,None]
                        self.opaque+=1
                        try:result=original(state,weight,bias,ids)
                        finally:self.opaque-=1
                        self.record('linear',[state,weight]+([] if bias is None else [bias])+[id_view],[result],
                                    name=name,indexed_source=True)
                        with torch._C._DisableTorchDispatch():
                            self.events[-1]['_index_values']=[ids.detach()]
                        return result
                    return execute
                for module,name in ((progressive_triton,'indexed_linear'),(progressive,'mapped_linear')):
                    original_affine=getattr(module,name)
                    setattr(module,name,indexed_affine_capture(original_affine,'chorus.'+name))
                    self.saved.append((module,name,original_affine))
                original_center_distance=progressive_triton.center_distance
                def progressive_center_distance(state,center,mapping):
                    map_view=mapping[:,None]
                    self.opaque+=1
                    try:result=original_center_distance(state,center,mapping)
                    finally:self.opaque-=1
                    # The physical scalar vector is a [row,1] view, so a
                    # selected distance preserves that row's exact sources.
                    self.record('progressive_center_distance',[state,center,map_view],[result[:,None]],
                                name='chorus.center_distance')
                    with torch._C._DisableTorchDispatch():
                        self.events[-1]['_index_values']=[mapping.detach()]
                    return result
                progressive_triton.center_distance=progressive_center_distance
                self.saved.append((progressive_triton,'center_distance',original_center_distance))
                original_order=progressive.consumer_order
                def progressive_order(keys):
                    self.opaque+=1
                    try:result=original_order(keys)
                    finally:self.opaque-=1
                    self.record('progressive_sort',[keys],[result],name='chorus.consumer_order')
                    return result
                progressive.consumer_order=progressive_order
                self.saved.append((progressive,'consumer_order',original_order))
                original_difference=progressive_triton.difference
                def progressive_difference(current,pilot,mapping,ids,begin):
                    map_view=mapping[:,None];id_view=ids[:,None]
                    pilot_view=pilot[:,begin:begin+current.shape[1]]
                    self.opaque+=1
                    try:result=original_difference(current,pilot,mapping,ids,begin)
                    finally:self.opaque-=1
                    self.record('progressive_difference',[current,pilot_view,map_view,id_view],[result],
                                name='chorus.progressive_difference',begin=0,hidden_column=begin)
                    with torch._C._DisableTorchDispatch():
                        self.events[-1]['_index_values']=[ids.detach(),mapping[ids].detach()]
                    return result
                progressive_triton.difference=progressive_difference
                self.saved.append((progressive_triton,'difference',original_difference))
                original_update=progressive_triton.update
                def progressive_update(output,contribution,ids,materialize=True,half_observation=True):
                    id_view=ids[:,None]
                    self.opaque+=1
                    try:result=original_update(output,contribution,ids,materialize,half_observation)
                    finally:self.opaque-=1
                    self.record('progressive_update',[output,contribution,id_view],[output,result],
                                name='chorus.progressive_update',materialize=materialize,half_observation=half_observation)
                    self.events[-1]['_index_values']=[ids.detach()]
                    return result
                progressive_triton.update=progressive_update
                self.saved.append((progressive_triton,'update',original_update))
                original_observe=progressive_triton.observe
                def progressive_observe(*args,**kwargs):
                    inputs=list(args[:10])
                    for j in (1,2,8,9):inputs[j]=inputs[j][:,None]
                    self.opaque+=1
                    try:result=original_observe(*args,**kwargs)
                    finally:self.opaque-=1
                    names=('window','bins','depth','depths','tolerance','force_full')
                    metadata=dict(zip(names,args[10:]));metadata.update(kwargs)
                    outputs=[result[0],result[1][:,None],args[3],args[4]]
                    self.record('progressive_observe',inputs,outputs,
                                name='chorus.progressive_observe',**metadata)
                    with torch._C._DisableTorchDispatch():
                        self.events[-1]['_index_values']=[args[1].detach(),args[2][args[1]].detach()]
                    return result
                progressive_triton.observe=progressive_observe
                self.saved.append((progressive_triton,'observe',original_observe))
            # Native SubMConv stores only half the pair counts; its central
            # matrix multiplication and opposite offsets are implicit. The
            # generic recorder's count loop misses these real dependencies.
            import spconv.pytorch as spconv
            from .convolution import native_submanifold_map
            for name,module in self.model.named_modules():
                if not isinstance(module,spconv.SubMConv3d):continue
                def complete_rulebook(mod,args,out,name=name):
                    info=out.indice_dict[mod.indice_key]
                    if hasattr(info,'pair_fwd'):return
                    with torch._C._DisableTorchDispatch():
                        mapping=native_submanifold_map(info.indice_pairs.cpu().numpy(),
                            info.indice_pair_num.cpu().numpy(),len(out.features))
                    self.maps[name]['mapping']=torch.from_numpy(mapping)
                    self.maps[name]['native_submanifold_expanded']=True
                self.handles.append(module.register_forward_hook(complete_rulebook))
            if family!='EmbodiedSplat':return
            import MinkowskiEngine as ME
            from measure_embodiedsplat_native import map_rows
            for name,module in self.model.named_modules():
                if not isinstance(module,(ME.MinkowskiConvolution,ME.MinkowskiConvolutionTranspose)) or module.use_mm or hasattr(module,'equivalent_spconv'):continue
                def ordered_rulebook(mod,args,out,name=name):
                    gen=mod.kernel_generator
                    pairs=args[0].coordinate_manager.kernel_map(args[0].coordinate_map_key,
                        out.coordinate_map_key,stride=gen.kernel_stride,kernel_size=gen.kernel_size,
                        dilation=gen.kernel_dilation,region_type=gen.region_type,is_transpose=mod.is_transpose)
                    from .convolution import ordered_kernel_pairs
                    self.maps[name]['offsets']=[torch.from_numpy(pair) for pair in
                        ordered_kernel_pairs({k:v.detach().cpu().numpy() for k,v in pairs.items()},gen.kernel_volume)]
                    self.maps[name]['kernel_offset_ids_preserved']=True
                self.handles.append(module.register_forward_hook(ordered_rulebook))
            original_insert=ME.CoordinateManager.insert_and_map
            def insert(manager,coordinates,tensor_stride=1,string_id=''):
                if self.opaque:return original_insert(manager,coordinates,tensor_stride,string_id)
                self.opaque+=1
                try:result=original_insert(manager,coordinates,tensor_stride,string_id)
                finally:self.opaque-=1
                coordinate_key,(unique,inverse)=result
                name=f'coordinate_insert_{self.call_counts["coordinate_insert"]}'
                self.call_counts['coordinate_insert']+=1
                with torch._C._DisableTorchDispatch():
                    targets=coordinates[unique.long()].detach().cpu()
                self.maps[name]=dict(target_coordinates=targets)
                stages=max(1,(len(coordinates)-1).bit_length())
                self.record('coordinate_map',[coordinates],[unique,inverse],name=name,map_key=name,
                    geometry_scalar_ops=len(coordinates)*(4*stages+4))
                return coordinate_key,(unique,inverse)
            ME.CoordinateManager.insert_and_map=insert
            self.saved.append((ME.CoordinateManager,'insert_and_map',original_insert))
            original_rows=ME.SparseTensor._batchwise_row_indices
            def batch_rows(value):
                if self.opaque:return original_rows.fget(value)
                self.opaque+=1
                try:result=original_rows.fget(value)
                finally:self.opaque-=1
                name=f'coordinate_decompose_{self.call_counts["coordinate_decompose"]}'
                self.call_counts['coordinate_decompose']+=1
                with torch._C._DisableTorchDispatch():
                    coordinates=value.C
                    targets=[coordinates[rows.long()].detach().cpu() for rows in result]
                self.maps[name]=dict(target_coordinates=targets)
                stages=max(1,(len(coordinates)-1).bit_length())
                self.record('coordinate_decomposition',[coordinates],result,name=name,map_key=name,
                    geometry_scalar_ops=len(coordinates)*(4*stages+4))
                return result
            ME.SparseTensor._batchwise_row_indices=property(batch_rows)
            self.saved.append((ME.SparseTensor,'_batchwise_row_indices',original_rows))
            original=ME.SparseTensor.features_at_coordinates
            trace=self
            def gather(value,coordinates):
                trace.opaque+=1
                try:result=original(value,coordinates)
                finally:trace.opaque-=1
                with torch._C._DisableTorchDispatch():
                    query=coordinates.detach().cpu().numpy()
                    if not np.array_equal(query,np.rint(query)):
                        raise NotImplementedError('Non-integer Minkowski interpolation needs weighted edges')
                    mapping=map_rows(query.astype(np.int32),value.C.cpu().numpy().astype(np.int32),value.tensor_stride)
                trace.record('aten',[value.F,coordinates],[result],op='aten.index_select.default',
                             name='Minkowski.features_at_coordinates')
                trace.events[-1]['_index_values']=[torch.as_tensor(mapping,dtype=torch.int64)]
                return result
            ME.SparseTensor.features_at_coordinates=gather
            self.saved.append((ME.SparseTensor,'features_at_coordinates',original))
    return Trace


def dump_trace(trace,result,destination,tag,metadata,checkpoints):
    """Export opaque maps and integer provenance for CPU timing consumers."""
    import pickle
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    indices={}
    for i,event in enumerate(trace.events):
        if '_index_values' in event:indices[i]=event.pop('_index_values')
    maps=dict(indices=indices,maps=trace.maps,checkpoints=checkpoints)
    if getattr(trace,'channel_checkpoints',None):maps['channel_checkpoints']=trace.channel_checkpoints
    if getattr(trace,'coverage_checkpoints',None):
        maps['coverage_checkpoints']=trace.coverage_checkpoints+checkpoints
    torch.save(maps,destination/f'{tag}_maps.pt')
    def portable(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().numpy()
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,tuple):return tuple(portable(v) for v in value)
        return value
    (destination/f'{tag}_maps.pkl').write_bytes(pickle.dumps(portable(maps),protocol=5))
    record=dict(metadata,events=trace.events,storages=list(trace.storages.values()),
                output_shape=list(result.shape),measured_device=torch.cuda.get_device_name())
    (destination/f'{tag}.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def dump_native_geometry(model,entry,destination,tag):
    """Export actual unclipped Gaussian support in the encoded native order."""
    if tag.startswith('EmbodiedSplat_'):
        source=WORKLOADS/'results/embodiedsplat_native_measurement/geometry'/f'{tag}.pt'
        support=torch.load(source,map_location='cpu',weights_only=False)
        record=model._garda_native[0]
        if not torch.equal(support['means'],record['coordinates']):
            raise ValueError('EmbodiedSplat support does not match the native encoded row order')
        if len(support['covariances'])!=len(record['coordinates']) or len(support['opacities'])!=len(record['coordinates']):
            raise ValueError('EmbodiedSplat support fields have inconsistent row counts')
        path=Path(destination)/f'{tag}_geometry.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,means=support['means'].numpy(),
            covariances=support['covariances'].numpy(),opacities=support['opacities'].numpy())
        return path
    import scenesplat_runtime as scene
    from .workloads import select_fragment
    settings=json.loads((WORKLOADS/'results/scenesplat_fp16.json').read_text())['settings']
    physical=scene.prepare_scene(WORKLOADS/entry['local'],settings['seed'],clip_scale=False)
    scope=model._garda_input_scope['input_scope']
    chunk,indices,_=select_fragment(scene,physical,settings,scope)
    model_chunk=model._garda_chunk;model_indices=model._garda_fragment_indices
    if not np.array_equal(chunk['coord'][indices],model_chunk['coord'][model_indices]):
        raise ValueError('Physical Gaussian attributes do not match the encoded native row order')
    path=Path(destination)/f'{tag}_geometry.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,coord=model._garda_inputs['coord'].detach().cpu().numpy(),feat=chunk['feat'][indices])
    return path


@torch.inference_mode()
def capture_progressive(dataset,variant,query_proxy=False,destination='results/raw/progressive_timing',parent_window=None,policy=None,deferred_output=False,channel_chunk=512,*,head_setup=None):
    """Capture a causal candidate execution; no future-derived stop schedule."""
    from chorus_runtime import official_modules,texts
    from .numerical import metrics
    from .progressive import progressive_projector,calibrated_lut,observation_root
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model,call,context,entry=load_case('Chorus',dataset,0);text=texts(dataset).cuda()
        root=observation_root('results/raw/progressive_actual',True,query_proxy,deferred_output,channel_chunk,
                              bool(policy and policy.get('channel_state_fp16',False)),
                              bool(policy and policy.get('channel_query_transposed',False)),
                              bool(policy and policy.get('channel_center_consumers',False)),
                              bool(policy and policy.get('channel_center_fused_distance',False)),
                              bool(policy and policy.get('channel_indexed_linear',False)),
                              (policy or {}).get('channel_context_level',2),
                              (policy or {}).get('channel_checkpoint_growth',1))
        lut=calibrated_lut(dataset,variant,root,policy,channel_chunk)
        with context:
            reference=call().clone()
            with ExitStack() as stack:
                preparation=trace_type('Chorus')(model,None,official_modules())
                preparation.stage='query_preparation'
                with preparation:
                    head=stack.enter_context(progressive_projector(model,text,lut,variant=='A_ancestor',fused_software=True,
                                               query_proxy=query_proxy,parent_window=parent_window,deferred_output=deferred_output,chunk=channel_chunk))
                    composition=(head_setup(model,head,text) if head_setup is not None else None)
                roots=[preparation.spec(v) for v in list(head.buffers())+list(head.parameters()) if v.is_cuda]
                dump_trace(preparation,head.second_weight,destination,f'Chorus_{dataset}_s0_preparation',dict(
                    model='Chorus',dataset=dataset,variant=variant,preparation_outputs=roots,
                    scope='One query/model preparation; all generated buffers and reordered weights required'),[])
                expected=call().clone()
                trace=trace_type('Chorus')(model,None,official_modules())
                trace.attach();trace.attach_stages('Chorus')
                try:
                    with trace:result=call()
                finally:trace.close()
                error=float((result.float()-expected.float()).abs().max())
                if error:raise ValueError(f'Instrumented candidate changed its output: {error}')
                quality=metrics(result,reference,text)
                checkpoint=dict(name='final',spec=trace.spec(result),mapping=np.arange(len(result)),after_event=len(trace.events)-1)
                geometry_path=dump_native_geometry(model,entry,destination,f'Chorus_{dataset}_s0')
                record=dump_trace(trace,result,destination,f'Chorus_{dataset}_s0',dict(
                    model='Chorus',dataset=dataset,index=0,entry=entry,variant=variant,query_proxy=query_proxy,parent_window=parent_window,
                    input_metadata=model._garda_input_scope,geometry_file=geometry_path.name,
                    geometry_scale='unclipped physical dataset scale',
                    predictor_policy=lut.config,
                    composition=composition,
                    risk_calibration=getattr(lut,'risk_calibration',None),
                    deferred_output=deferred_output,
                    channel_chunk=channel_chunk,
                    output_max_abs_error=error,quality=quality,active_rows_per_checkpoint=head.executed_rows,
                    completed_consumers=head.pilots,response_consumers=head.feedback_rows,
                    scope='Complete causal 3D encoding; actual online decisions; native coverage domains available',
                    query_preparation_in_trace=False,
                    query_preparation_trace=f'Chorus_{dataset}_s0_preparation.json'),[checkpoint])
                print(dataset,variant,'causal timing trace',len(trace.events),'events',quality,flush=True)
                return record
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


@torch.inference_mode()
def capture(family, dataset, index, destination, mixed_extended=False, wide_kernel=False,query_fp16=False,query_transposed=False,output_fp16=False,query_factorized=False,embodied_native_previous=False,terminal_latent_query=False,
            *,input_scope='chunk-first',trace_only=False,ancestor_stripe_columns=None,ancestor_full=False,query_partitioned=False,output_partitioned=False,tail_rank=None,export_stage_vectors=False,chorus_readouts=None):
    if chorus_readouts is not None and (family!='Chorus' or trace_only):
        raise ValueError('Alternate Chorus readouts require a Chorus diagnostic capture')
    if input_scope=='scene-first' and not trace_only:
        raise ValueError('Scene-first currently requires trace-only; bounded large-input diagnostic capture is not implemented')
    if ancestor_stripe_columns is not None and (not trace_only or family!='Chorus' or ancestor_stripe_columns<1):
        raise ValueError('Ancestor channel stripes require a positive width and Chorus trace-only capture')
    if ancestor_full and (family!='Chorus' or ancestor_stripe_columns is not None):
        raise ValueError('Full ancestor projection requires Chorus without a stripe width')
    ancestor=ancestor_full or ancestor_stripe_columns is not None
    if tail_rank is not None and (family!='Chorus' or mixed_extended or wide_kernel):
        raise ValueError('Ranked tail currently requires Chorus without other public arithmetic variants')
    if trace_only and (mixed_extended or wide_kernel or query_fp16 or query_transposed or output_fp16):
        raise ValueError('Trace-only capture records unmodified native execution, without diagnostics or precision variants')
    if trace_only and family=='EmbodiedSplat':
        raise ValueError('Trace-only capture currently supports native SceneSplat/Chorus inputs')
    if trace_only and export_stage_vectors:
        raise ValueError('Stage vector export requires diagnostic capture')
    if query_transposed and not query_fp16:
        raise ValueError('Transposed query capture requires the FP16 query path')
    if query_factorized and (family!='EmbodiedSplat' or not query_fp16):
        raise ValueError('Factorized queries require EmbodiedSplat FP16 control')
    if embodied_native_previous and family!='EmbodiedSplat':
        raise ValueError('The native preceding readout is specific to EmbodiedSplat')
    if terminal_latent_query and (family!='EmbodiedSplat' or not query_fp16):
        raise ValueError('Terminal latent queries require EmbodiedSplat FP16 control')
    if query_partitioned and (family!='Chorus' or not query_transposed):
        raise ValueError('Partitioned query currently requires Chorus and transposed FP16 composition')
    if output_partitioned and not (query_partitioned and output_fp16):
        raise ValueError('Partitioned output requires partitioned Chorus queries and FP16 early output')
    if output_fp16 and family not in ('SceneSplat','Chorus','EmbodiedSplat'):
        raise ValueError('FP16 early output requires a captured semantic readout')
    if (mixed_extended or wide_kernel) and family!='EmbodiedSplat':
        raise ValueError('Extended precision is only supported for EmbodiedSplat')
    if wide_kernel and not mixed_extended:
        raise ValueError('Wide precision requires extended precision')
    Trace=trace_type(family)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tag = f'{family}_{dataset}_s{index}'
    lock = (WORKLOADS / 'results/context_reload_v1/.gpu_counter.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, call, context, entry = load_case(family, dataset, index, input_scope=input_scope)
    official = None
    if family != 'EmbodiedSplat':
        from scenesplat_runtime import official_modules
        if family == 'Chorus':
            from chorus_runtime import official_modules
        official = official_modules()
    trace = Trace(model, None, official)
    if trace_only:
        trace.storage_handles=WeakStorageOwners()
    checkpoints, parents, handles = [], {}, []
    stem = {}
    if family == 'EmbodiedSplat':
        from measure_embodiedsplat_native import map_rows
        record, coordinates, features, inverse = model._garda_native
        gaussian_coords = coordinates[inverse].cpu().numpy().astype(np.int32)
        def capture_stem(m,a):
            out=a[0]
            with torch._C._DisableTorchDispatch():
                mapping=map_rows(gaussian_coords,out.C.cpu().numpy().astype(np.int32),out.tensor_stride)
                value=out.F.detach().clone()
            stem.update(spec=trace.spec(out.F),value=value,mapping=mapping)
        handles.append(model.unet3d.conv1p1s2.register_forward_pre_hook(capture_stem))
        def checkpoint(module, args, out, name):
            with torch._C._DisableTorchDispatch():
                mapping = map_rows(gaussian_coords, out.C.cpu().numpy().astype(np.int32), out.tensor_stride)
                value = out.F.detach().clone()
            checkpoints.append(dict(name=name, spec=trace.spec(out.F), value=value, mapping=mapping,
                                    after_event=len(trace.events)-1))
        for stage in range(1, 9):
            for block in range(2):
                name = f'unet3d.block{stage}.{block}'
                handles.append(model.get_submodule(name).register_forward_hook(
                    lambda m,a,o,name=name:checkpoint(m,a,o,name)))
        xyz = record['coordinates'].numpy()
    else:
        inputs = model._garda_inputs
        xyz = inputs['coord'].cpu().numpy()
        for name, module in model.named_modules():
            if re.fullmatch(r'(backbone\.)?enc.enc[1-4]', name):
                level = int(name[-1])
                def parent(m,a,o,level=level):
                    with torch._C._DisableTorchDispatch():
                        parents[level] = o.pooling_inverse.detach().cpu().numpy().copy()
                handles.append(module.register_forward_hook(parent))
            if re.fullmatch(r'(backbone\.)?(enc.enc|dec.dec)\d+.block\d+', name):
                def checkpoint(m,a,o,name=name):
                    cp=dict(name=name, spec=trace.spec(o.feat), after_event=len(trace.events)-1)
                    if not trace_only:
                        with torch._C._DisableTorchDispatch():
                            cp['value'] = o.feat.detach().clone()
                    checkpoints.append(cp)
                handles.append(module.register_forward_hook(checkpoint))
    # Register opaque hooks before checkpoint hooks: replace the order for parent
    # blocks is unnecessary because each block's final ATen write is already seen.
    with ExitStack() as stack:
        stack.enter_context(context)
        quality_reference=None
        preparation_name=None
        if tail_rank is not None:
            from .factorized import ranked_language_tail
            quality_reference=call().clone()
            # Query-independent frozen weight compilation is outside scene
            # encoding, as is checkpoint loading. Both runtime GEMMs remain
            # in the trace; the original result remains the quality target.
            stack.enter_context(ranked_language_tail(model,tail_rank))
            trace=Trace(model,None,official)
            if trace_only:trace.storage_handles=WeakStorageOwners()
        if ancestor:
            from .factorized import ancestor_projector
            # A composed rank-tail candidate must still compare with the
            # original unmodified complete output, not the installed rank.
            if quality_reference is None:quality_reference=call().clone()
            preparation=Trace(model,None,official)
            preparation.stage='model_preparation'
            with preparation:
                affine=stack.enter_context(ancestor_projector(model,projection_dtype=torch.float16,
                    use_fused=True,inline_fine=True,stripe_columns=ancestor_stripe_columns))
            preparation_name=f'{tag}_preparation.json'
            outputs=[preparation.spec(value) for value in list(affine.buffers())+list(affine.parameters())]
            dump_trace(preparation,affine.constant,destination,Path(preparation_name).stem,dict(
                model=family,dataset=dataset,preparation_outputs=outputs,
                scope='One actual frozen ancestor parameter preparation'),[])
            trace=Trace(model,None,official)
            trace.storage_handles=WeakStorageOwners()
        if mixed_extended:
            quality_reference=call().clone()
            from .precision import submanifold_fp16
            stack.enter_context(submanifold_fp16(model,extended=True,wide_kernel=wide_kernel))
            # Register actual converted parameters, not pre-conversion weights.
            trace=Trace(model,None,official)
        for _ in range(2):
            reference = call().clone()
        checkpoints.clear(); parents.clear()
        trace.attach(); trace.attach_stages(family)
        try:
            with trace:
                result = call()
        finally:
            trace.close()
            for handle in handles:
                handle.remove()
        max_error = float((result.float()-reference.float()).abs().max())
        if not mixed_extended and max_error > (2e-5 if family == 'EmbodiedSplat' else 0):
            raise RuntimeError(f'Instrumented output changed: {max_error}')
    if family != 'EmbodiedSplat':
        leafmap = {0:np.arange(len(xyz), dtype=np.int32)}
        for level in sorted(parents):
            leafmap[level] = parents[level][leafmap[level-1]]
        for cp in checkpoints:
            level = int(re.search(r'(?:enc|dec)\d+', cp['name'])[0][-1])
            cp['mapping'] = leafmap[level]
    input_metadata=getattr(model,'_garda_input_scope',dict(input_scope='native-cache'))
    candidate_metadata={}
    if tail_rank is not None:
        from .numerical import metrics
        from chorus_runtime import texts
        quality=metrics(result,quality_reference,texts(dataset).cuda())
        if not quality['passes_thresholds']:raise ValueError('Ranked tail failed original complete-output quality')
        candidate_metadata=dict(variant='common_ranked_tail',tail_rank=tail_rank,quality=quality,
            weight_preparation='Offline query-independent SVD; both factor GEMMs in encoding')
    if ancestor:
        from .numerical import metrics,bounded_metrics
        from chorus_runtime import texts
        quality=(bounded_metrics if trace_only else metrics)(result,quality_reference,texts(dataset).cuda())
        if not quality['passes_thresholds']:
            raise ValueError('Common ancestor candidate failed native output quality')
        candidate_metadata.update(variant='common_ancestor_full' if ancestor_full else 'common_ancestor_stripes',
            ancestor_stripe_columns=ancestor_stripe_columns,quality=quality,
            execution_shape=affine.execution_shape,query_preparation_trace=preparation_name,
            query_preparation_in_trace=False,
            quality_reference='Original complete native output, including fallback observations')
    if trace_only:
        checkpoints.append(dict(name='final',spec=trace.spec(result),
            mapping=np.arange(len(result),dtype=np.int32),after_event=len(trace.events)-1))
        geometry_path=dump_native_geometry(model,entry,destination,tag)
        metadata=dict(model=family,dataset=dataset,index=index,
            entry=entry,output_max_abs_error=max_error,input_metadata=input_metadata,
            diagnostic_observations=False,geometry_file=geometry_path.name,
            geometry_scale='unclipped physical dataset scale',storage_identity_lifetime='weak StorageImpl owners',
            scope='Complete native first-fragment 3D encoding; no diagnostic observations')
        metadata.update(candidate_metadata)
        record=dump_trace(trace,result,destination,tag,metadata,checkpoints)
        print(tag,'captured native execution',len(trace.events),'events',input_metadata,flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        return record
    # Frozen independent-scene alignment, charged as actual online diagnostic
    # transforms by the cycle model. No future model value enters a readout.
    if family == 'SceneSplat':
        from profile_scenesplat import texts
        weights = torch.load(WORKLOADS/'results/readouts_aggregation_matched.pt', weights_only=False)['weights']
        weights.update(torch.load(WORKLOADS/'results/readouts_fine_decoder.pt', weights_only=False)['weights'])
        prefix = torch.load(WORKLOADS/'results/readouts_block_prefixes.pt', weights_only=False)
        text = texts(dataset).cuda()
    elif family == 'Chorus':
        from chorus_runtime import texts
        weights = torch.load(chorus_readouts or WORKLOADS/'results/chorus_readouts.pt', weights_only=False)
        text = texts(dataset).cuda()
    else:
        selected = torch.load(WORKLOADS/'results/embodiedsplat_native_measurement/diagnostics.pt', weights_only=False)[dataset]['three_d']
        text = torch.load(WORKLOADS/'results/embodiedsplat_native_measurement/texts.pt', weights_only=False)[dataset]['features'].cuda()
        bank = [dict(spec=trace.spec(features),value=features,mapping=inverse.cpu().numpy())]
    final = F.normalize(result.float(),dim=-1)
    reference_final=F.normalize(quality_reference.float(),dim=-1) if quality_reference is not None else final
    final_scores = reference_final @ text.T
    target = final_scores.argmax(-1)
    previous = None
    previous_stage = None
    observations = []
    recipe_checks = []
    exported_vectors=[];exported_checkpoints=[];terminal_control=None
    stage_last = {}
    if family == 'Chorus':
        for j, cp in enumerate(checkpoints):
            stage_last[int(re.search(r'enc(\d+)',cp['name'])[1])] = j
    for j, cp in enumerate(checkpoints):
        x = cp['value'].float()
        mapping = torch.as_tensor(cp['mapping'], device='cuda').long()
        cp['readout_inputs'] = [cp['spec']]
        cp['readout_maps'] = [cp['mapping']]
        extra_flops = 0
        control=None
        if family == 'SceneSplat':
            key = f'b{j+1:02d}'
            if j == len(checkpoints)-1:
                language = F.normalize(x,dim=1)[mapping]
                cp['readout_shape'] = [x.shape[1], x.shape[1]]
                cp['readout_identity'] = True
            else:
                stage = re.search(r'(enc|dec)\d+', cp['name'])[0]
                probe = 'dec0_block0' if j == len(checkpoints)-2 else stage
                w = prefix['weights'][key] if key in prefix['weights'] and prefix['selection'][key]['use_prefix'] else weights[probe]
                w = w.cuda()
                language = F.normalize(F.pad(F.normalize(x,dim=1),(0,1),value=1)@w,dim=1)[mapping]
                cp['readout_shape'] = list(w.shape)
                from .readout import affine_recipe,affine_language
                cp['readout_recipe']=affine_recipe(family,1)
                # The native probe runs on canonical rows before expanding
                # leaves. Keep that GEMM shape: leaf expansion can change
                # FP32 reduction rounding and flip a nearly tied argmax.
                recipe_parts=[cp['value']]
                checked=affine_language(recipe_parts,w,cp['readout_recipe'])[mapping]
        elif family == 'Chorus':
            key = f'b{j+1:02d}'
            level = int(re.search(r'enc(\d+)',cp['name'])[1])
            kind = weights['selection'][key]['kind']
            parts = []
            if kind.startswith('prefix'):
                for level_before in range(level):
                    earlier = checkpoints[stage_last[level_before]]
                    parts.append(earlier['value'].float()[torch.as_tensor(earlier['mapping'],device='cuda').long()])
                    cp['readout_inputs'].append(earlier['spec'])
                    cp['readout_maps'].append(earlier['mapping'])
            parts.append(x[mapping]);x=torch.cat(parts,dim=1)
            if kind.endswith('centered'):x=x-x.mean(1,keepdim=True)
            w=weights['weights'][f'{key}/{kind}'].cuda()
            language=F.normalize(F.pad(F.normalize(x,dim=1),(0,1),value=1)@w,dim=1)
            cp['readout_shape']=list(w.shape)
            from .readout import affine_recipe,affine_language
            cp['readout_recipe']=affine_recipe(family,len(cp['readout_inputs']),kind.endswith('centered'))
            inputs=[checkpoints[j]]+[checkpoints[stage_last[k]] for k in range(level)] if kind.startswith('prefix') else [cp]
            recipe_parts=[a['value'][torch.as_tensor(a['mapping'],device='cuda').long()] for a in inputs]
            checked=affine_language(recipe_parts,w,cp['readout_recipe'])
        else:
            item=selected['readouts'][j];kind=item['kind'];head=selected['head']
            if embodied_native_previous and j==len(checkpoints)-3:
                # The preceding decoder stage already has the terminal 96-D
                # width. Reuse the frozen model head as a low-cost trajectory
                # observation, independent of labels and query vocabulary.
                kind='native'
            current=x[mapping]
            if kind=='native':
                latent=current@head['kernel'].cuda()
                language=F.normalize(F.relu(latent)@head['language_weight'].cuda()+head['language_bias'].cuda(),dim=1)
                if query_fp16:
                    from .diagnostics import half_query
                    control=half_query(F.relu(latent),head['language_weight'].cuda(),text,head['language_bias'].cuda(),transposed=query_transposed)
                    cp['query_bias']=True
                cp['readout_shape']=[current.shape[1],64]
                extra_flops=2*64*language.shape[1]
            else:
                if kind.startswith('teacher'):
                    raise RuntimeError('Upstream semantic teacher is not a 3D early checkpoint')
                norm=(lambda a:a) if kind.startswith('latent') else (lambda a:F.normalize(a,dim=1))
                parts=[norm(current)]
                if kind.endswith('prefix'):
                    for earlier in [stem]+bank:
                        part=earlier['value'].float()[torch.as_tensor(earlier['mapping'],device='cuda').long()]
                        parts.append(norm(part))
                        cp['readout_inputs'].append(earlier['spec'])
                        cp['readout_maps'].append(earlier['mapping'])
                xx=torch.cat(parts,1)
                if not kind.startswith('latent'):xx=F.normalize(xx,dim=1)
                w=item['weight'].cuda()
                latent=F.pad(xx,(0,1),value=1)@w
                language=F.normalize(F.relu(latent)@head['language_weight'].cuda()+head['language_bias'].cuda(),dim=1) if kind.startswith('latent') else F.normalize(latent,dim=1)
                if query_fp16:
                    from .diagnostics import half_query
                    source=F.pad(xx,(0,1),value=1)
                    if query_factorized and not kind.startswith('latent'):
                        u,s,vh=torch.linalg.svd(w.float(),full_matrices=False)
                        tolerance=s[0]*max(w.shape)*torch.finfo(s.dtype).eps
                        rank=int((s>tolerance).sum())
                        if rank<w.shape[0]:
                            left=u[:,:rank]*s[:rank]
                            right=vh[:rank]
                            from .diagnostics import factorized_half_query
                            control=factorized_half_query(source,left,right,text)
                            cp['query_factor_shapes']=[list(left.shape),list(right.shape)]
                            cp['query_factor_rank']=rank
                            cp['query_factor_max_abs_error']=float((left@right-w.float()).abs().max())
                        else:control=half_query(source,w,text,transposed=query_transposed)
                    else:
                        control=(half_query(F.relu(latent),head['language_weight'].cuda(),text,head['language_bias'].cuda(),transposed=query_transposed)
                                 if kind.startswith('latent') else half_query(source,w,text,transposed=query_transposed))
                    cp['query_bias']=kind.startswith('latent')
                cp['readout_shape']=list(w.shape)
                if kind.startswith('latent'):extra_flops=2*64*language.shape[1]
            if j%2==1:bank.append(cp)
        if query_fp16 and family in ('SceneSplat','Chorus'):
            from .diagnostics import half_query
            if cp.get('readout_identity',False):
                # Query normalization cancels the positive language norm.
                # Match the finite identity path: raw checkpoint -> half mm.
                control=(cp['value'].half()@text.T.half()).float()[mapping]
            else:
                from .readout import affine_source
                if query_partitioned:
                    from .diagnostics import partitioned_query
                    order=cp['readout_recipe']['input_order']
                    control=partitioned_query([inputs[k]['value'] for k in order],
                                              [inputs[k]['mapping'] for k in order],w,text,
                                              centered=cp['readout_recipe']['centered'])
                    cp['query_partitioned']=True
                else:control=half_query(affine_source(recipe_parts,cp['readout_recipe']),w,text,transposed=query_transposed)
                if family=='SceneSplat':control=control[mapping]
        scores=language@text.T
        if 'readout_recipe' in cp:
            from .readout import recipe_check
            recipe_checks.append((cp,recipe_check(language,checked,text)))
        top=scores.topk(2,dim=1)
        change=torch.zeros(len(language),device='cuda') if previous is None else (language-previous).abs().sum(1)
        stage_name=cp['name'].rsplit('.block',1)[0] if family!='EmbodiedSplat' else cp['name'].rsplit('.',1)[0]
        next_stage=(checkpoints[j+1]['name'].rsplit('.block',1)[0] if family!='EmbodiedSplat' else checkpoints[j+1]['name'].rsplit('.',1)[0]) if j+1<len(checkpoints) else None
        cp['stage_end']=stage_name!=next_stage
        stage_change=torch.zeros(len(language),device='cuda')
        if cp['stage_end']:
            if previous_stage is not None:stage_change=(language-previous_stage).abs().sum(1)
            previous_stage=language
        observations.append(dict(step_l1=change,stage_step_l1=stage_change,margin=top.values[:,0]-top.values[:,1],
                                 winner=top.indices[:,0], scores=scores,
                                 cosine=(language*reference_final).sum(1),
                                 squared_error=(language-reference_final).square().mean(1),
                                 regret=final_scores.max(1).values-final_scores.gather(1,top.indices[:,:1]).squeeze(1)))
        if query_fp16:
            observations[-1]['control_scores']=control
            cp['query_fp16']=True
            if family=='EmbodiedSplat' and j==len(checkpoints)-1:
                terminal_control=control
        if output_fp16:
            # Keep the FP32 language/scores for completed-pilot feedback and
            # frozen policy labels. Only the actually emitted early output
            # changes precision, with separate measured quality fields.
            half_language=language
            if not cp.get('readout_identity',False):
                if family=='EmbodiedSplat':
                    if kind=='native' or kind.startswith('latent'):
                        source=F.pad(F.relu(latent),(0,1),value=1).half()
                        weight=torch.cat((head['language_weight'].cuda(),
                                          head['language_bias'].cuda()[None]),dim=0).half()
                        half_language=F.normalize((source@weight).float(),dim=1)
                    else:
                        output_source=F.pad(xx,(0,1),value=1)
                        half_language=F.normalize((output_source.half()@w.half()).float(),dim=1)
                    cp['output_fp16']=True
                elif output_partitioned:
                    from .diagnostics import partitioned_language
                    order=cp['readout_recipe']['input_order']
                    half_language=partitioned_language([inputs[k]['value'] for k in order],
                        [inputs[k]['mapping'] for k in order],w,centered=cp['readout_recipe']['centered'])
                    cp['output_partitioned']=True
                else:half_language=affine_language(recipe_parts,w,cp['readout_recipe'],output_fp16=True)
                if family!='EmbodiedSplat':
                    if family=='SceneSplat':half_language=half_language[mapping]
                    cp['output_fp16']=True
            half_winner=(half_language@text.T).argmax(1)
            observations[-1].update(output_winner=half_winner,
                output_cosine=(half_language*reference_final).sum(1),
                output_squared_error=(half_language-reference_final).square().mean(1),
                output_regret=final_scores.max(1).values-final_scores.gather(1,half_winner[:,None]).squeeze(1))
        if export_stage_vectors and cp['stage_end']:
            # Research artifact only. Capture the actual early-output path;
            # explicit FP16 storage is part of any later residual hypothesis.
            exported_vectors.append((half_language if output_fp16 else language).half())
            exported_checkpoints.append(j)
        previous=language
        cp['readout_extra_flops_per_row']=extra_flops
        cp['semantic_width']=language.shape[1]
    if recipe_checks:
        # Transfer the complete verification vector once, rather than sync
        # each checkpoint or Gaussian to the host.
        checks=torch.stack([value for _,value in recipe_checks]).cpu().numpy()
        for (cp,_),(error,agreement) in zip(recipe_checks,checks):
            cp['readout_recipe_verification']=dict(max_abs_error=float(error),class_agreement=float(agreement))
            from .readout import validate_recipe
            validate_recipe(cp,family)
    for cp in checkpoints:del cp['value']
    # Append exact final output as a guaranteed fallback for all models.
    final_checkpoint=dict(name='final',spec=trace.spec(result),mapping=np.arange(len(final),dtype=np.int32),
        after_event=len(trace.events)-1,readout_shape=[0,0],readout_inputs=[],readout_maps=[],
        semantic_width=final.shape[1],readout_extra_flops_per_row=0,readout_identity=True)
    if terminal_latent_query:
        candidates=[e['outputs'][0] for e in trace.events if e['stage']=='output_3d' and
                    e['outputs'] and len(e['outputs'][0]['shape'])==2 and
                    e['outputs'][0]['shape'][-1]==64]
        if not candidates or terminal_control is None:
            raise ValueError('Captured EmbodiedSplat terminal latent is unavailable')
        latent_spec=candidates[-1]
        final_checkpoint.update(readout_shape=[65,final.shape[1]],
            readout_inputs=[latent_spec],readout_maps=[checkpoints[-1]['mapping']],
            readout_identity=False,terminal_latent_query=True,query_fp16=True,query_bias=True)
    checkpoints.append(final_checkpoint)
    result_scores=final@text.T if quality_reference is not None else final_scores
    result_winner=result_scores.argmax(1)
    observations.append(dict(step_l1=(final-previous).abs().sum(1),stage_step_l1=(final-previous_stage).abs().sum(1),margin=result_scores.topk(2,dim=1).values.diff(dim=1).abs().squeeze(1),
                             winner=result_winner,scores=result_scores,cosine=(final*reference_final).sum(1),
                             squared_error=(final-reference_final).square().mean(1),
                             regret=final_scores.max(1).values-final_scores.gather(1,result_winner[:,None]).squeeze(1)))
    if query_fp16:
        observations[-1]['control_scores']=(terminal_control if terminal_latent_query else result_scores)
    if output_fp16:
        observations[-1].update({f'output_{k}':observations[-1][k]
                                 for k in ('winner','cosine','squared_error','regret')})
    # One compact transfer per field, no per-Gaussian GPU/CPU synchronization.
    arrays={k:torch.stack([r[k] for r in observations]).cpu().numpy() for k in observations[0]}
    arrays['xyz']=xyz
    arrays['target']=target.cpu().numpy()
    np.savez_compressed(destination/f'{tag}_observations.npz',**arrays)
    if export_stage_vectors:
        exported_vectors.append(final.half());exported_checkpoints.append(len(checkpoints)-1)
        # One stage-batched device transfer, no per-Gaussian synchronization.
        np.savez(destination/f'{tag}_stage_vectors.npz',
            language=torch.stack(exported_vectors).cpu().numpy(),
            checkpoint_indices=np.asarray(exported_checkpoints,np.int32),
            text=text.float().cpu().numpy(),reference_output=reference_final.cpu().numpy())
    geometry_path=dump_native_geometry(model,entry,destination,tag)
    metadata=dict(model=family,dataset=dataset,index=index,entry=entry,
        output_max_abs_error=max_error,input_metadata=input_metadata,
        diagnostic_observations=True,geometry_file=geometry_path.name,
        geometry_scale='unclipped physical dataset scale',
        scope='Complete dynamic 3D semantic encoding')
    metadata['diagnostic_query_fp16']=query_fp16
    metadata['diagnostic_factorized_query']=query_factorized
    metadata['embodied_native_previous']=embodied_native_previous
    metadata['terminal_latent_query']=terminal_latent_query
    metadata['diagnostic_transposed_composition']=query_transposed
    metadata['diagnostic_output_fp16']=output_fp16
    if export_stage_vectors:metadata['stage_vectors_export']='FP16 actual stage-end early outputs; FP32 original reference and query text; research artifact only'
    metadata['diagnostic_partitioned_query']=query_partitioned
    metadata['diagnostic_partitioned_output']=output_partitioned
    if chorus_readouts is not None:metadata['chorus_readouts']=str(Path(chorus_readouts).resolve())
    metadata.update(candidate_metadata)
    if mixed_extended:
        from .numerical import metrics
        metadata.update(variant='mixed_precision_extended_wide' if wide_kernel else 'mixed_precision_extended',
            extended=True,wide_kernel=wide_kernel,quality=metrics(result,quality_reference,text),
            quality_reference='Original FP32 complete output, including fallback observations')
        if not metadata['quality']['passes_thresholds']:raise ValueError('Mixed capture failed original FP32 quality')
    dump_trace(trace,result,destination,tag,metadata,checkpoints)
    print(tag,'captured',len(trace.events),'events',len(checkpoints),'checkpoints',flush=True)
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()
