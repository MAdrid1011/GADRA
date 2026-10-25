"""Versioned tensor dependencies and conservative per-row demand closure."""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pickle
from .protocol import SECTOR_BYTES,SLOT_BYTES,DEPENDENCY_GRANULE_BYTES


def rows(spec):
    return math.prod(spec['shape'][:-1]) if spec['shape'] else 1


def key(spec):
    return spec['storage'], spec['version']


def coverage_layout(maps,leaf_order,n,packing):
    """Lift witnessed Gaussian coverage order through a real parent mapping.

    Keep physical row chunks intact; coarse inputs and fine output identities
    stay unchanged. None means no captured geometry establishes this layout.
    """
    for cp in maps.get('checkpoints',[]):
        mapping=cp['mapping']
        if rows(cp['spec'])!=n or len(mapping)!=len(leaf_order):continue
        _,first=np.unique(mapping[leaf_order],return_index=True)
        candidate=mapping[leaf_order][np.sort(first)]
        chunks=candidate//packing
        _,first=np.unique(chunks,return_index=True)
        candidate=(chunks[np.sort(first),None]*packing+np.arange(packing)).ravel()
        candidate=candidate[candidate<n]
        if len(candidate)!=n:raise ValueError('Coverage layout does not cover every row')
        return candidate
    return None


class DemandPlan(list):
    """Rows and copy-forward ranges required by partial storage updates."""
    def __init__(self,live,carry=None):
        super().__init__(live)
        self.carry=carry or {}

    def minus(self,earlier):
        carry={}
        for index,(old,new,tokens) in self.carry.items():
            previous=earlier.carry.get(index)
            remaining=tokens if previous is None else np.setdiff1d(tokens,previous[2],assume_unique=True)
            if len(remaining):carry[index]=(old,new,remaining)
        return DemandPlan([np.setdiff1d(a,b,assume_unique=True) for a,b in zip(self,earlier)],carry)


class Graph:
    granule = DEPENDENCY_GRANULE_BYTES

    def __init__(self, path,*,execution_only=False):
        path=Path(path)
        self.path=path
        self.trace=json.loads(path.read_text())
        self.events=self.trace['events']
        for event in self.events:
            from .reductions import lower_row_statistics
            lower_row_statistics(event)
            if event.get('op')=='aten.linear.default':
                event['kind']='linear';event['lowered_from']='aten.linear.default'
            elif event.get('op') in ('aten.matmul.default','aten.mv.default') and len(event['inputs'])==2:
                left,right=event['inputs'];output=event['outputs'][0]
                # Vector-shaped products still execute real MACs. Canonical
                # row/column views preserve byte addresses and arithmetic.
                if len(left['shape'])==2 and len(right['shape'])==1:
                    right=dict(right,shape=[right['shape'][0],1],stride=[right['stride'][0],1])
                    output=dict(output,shape=[output['shape'][0],1],stride=[output['stride'][0],1])
                elif len(left['shape'])==1 and len(right['shape'])==2:
                    left=dict(left,shape=[1,left['shape'][0]],stride=[left['bytes']//left['element_bytes'],left['stride'][0]])
                    output=dict(output,shape=[1,output['shape'][0]],stride=[output['bytes']//output['element_bytes'],output['stride'][0]])
                if len(left['shape'])==len(right['shape'])==2:
                    event['lowered_from']=event['op'];event['op']='aten.mm.default'
                    event['inputs']=[left,right];event['outputs']=[output]
        portable=path.with_name(path.stem+'_maps.pkl')
        if portable.exists():self.maps=pickle.loads(portable.read_bytes())
        else:
            import torch
            self.maps=torch.load(path.with_name(path.stem+'_maps.pt'),map_location='cpu',weights_only=False)
        self.storage={s['id']:s for s in self.trace['storages']}
        self.initialize_dependency_granules()
        self.demand={}
        self.live=[]
        self.conservative=Counter()
        self.bases={}
        self.inplace_versions=set()
        self.inplace_feature_copy_versions=set()
        last_reads={key(s):i for i,e in enumerate(self.events) for s in e['inputs']}
        observed={key(spec) for cp in self.maps.get('checkpoints',[])
                  for spec in ([cp['spec']] if 'spec' in cp else [])+cp.get('readout_inputs',[])}
        # Explicit preparation roots are all observable results of a single
        # full execution. An older exported version must never be overwritten.
        observed.update(key(spec) for spec in self.trace.get('preparation_outputs',[]))
        fixed_work=(execution_only or self.trace.get('diagnostic_observations') is False or
                    (bool(self.trace.get('preparation_outputs')) and
                     not self.maps.get('checkpoints')))
        # Integer construction is mandatory and completed in the first phase.
        # Keep SSA renaming for any storage consumed by feature work: a later
        # feature phase may revisit an earlier version even after its last
        # lexical use. Pure metadata scratch has no such delayed consumers.
        index_only={s['storage'] for e in self.events for s in e['outputs'] if s['role']=='index'}
        for event in self.events:
            if any(s['role']=='feature' for s in event['outputs']):
                index_only.difference_update(s['storage'] for s in event['inputs'])
        offset=0
        for i,event in enumerate(self.events):
            for spec in event['inputs']+event['outputs']:
                if key(spec) not in self.bases:
                    old=next((s for s in event['inputs'] if s['storage']==spec['storage'] and
                              s['version']+1==spec['version'] and last_reads[key(s)]==i),None)
                    metadata_copy=(event.get('op')=='aten.copy_.default' and spec['storage'] in index_only)
                    # Execution-only traces have no deferred A feature phase.
                    # A native slice copy can therefore keep its physical
                    # destination if no old-version reader/observer survives.
                    # The actual masked write and RAW input wait remain paid.
                    feature_copy=(fixed_work and
                        event.get('op')=='aten.copy_.default' and spec['role']=='feature' and
                        old is not None and key(old) not in observed and
                        all(spec[field]==old[field] for field in ('address','shape','stride','element_bytes')))
                    if (event['kind'] in ('progressive_update','progressive_observe','indexed_store') or metadata_copy or feature_copy) and old is not None:
                        self.bases[key(spec)]=self.bases[key(old)]
                        self.inplace_versions.add((key(old),key(spec)))
                        if feature_copy:self.inplace_feature_copy_versions.add((key(old),key(spec)))
                        continue
                    self.bases[key(spec)]=offset
                    offset+=math.ceil(self.storage[spec['storage']]['bytes']/SLOT_BYTES)*SLOT_BYTES
        self.map_bases={}
        previous=[]
        for name,value in self.maps['maps'].items():
            if 'mapping' in value:
                mapping=np.asarray(value['mapping'])
                match=next((base for other,base in previous if other.shape==mapping.shape and np.array_equal(other,mapping)),None)
                if match is None:
                    match=offset;offset+=math.ceil(mapping.nbytes/SLOT_BYTES)*SLOT_BYTES;previous.append((mapping,match))
                self.map_bases[name]=match
            elif 'offsets' in value:
                # Minkowski rulebooks are real input/output index pairs.
                # Keep offset boundaries to stream exactly the selected edges.
                value['pair_byte_offsets']=np.cumsum([0]+[sum(a.nbytes for a in pair) for pair in value['offsets']])
                self.map_bases[name]=offset
                offset+=math.ceil(int(value['pair_byte_offsets'][-1])/SLOT_BYTES)*SLOT_BYTES

    def initialize_dependency_granules(self):
        """Logical byte ranges, independent of physical sector transfers.

        Wide aligned rows retain the compact sector bitmap. Skinny statistics,
        offsets and aliased views refine only their own storage. All versions
        use one unit, so inheritance tokens keep a stable byte interpretation.
        """
        self.dependency_granules={}
        views=[event['inputs']+event['outputs'] for event in self.events]
        for cp in self.maps.get('checkpoints',[]):
            views.append(([cp['spec']] if 'spec' in cp else [])+cp.get('readout_inputs',[]))
        for group in views:
            for spec in group:
                storage=spec['storage'];unit=self.dependency_granules.get(storage,self.granule)
                element=spec['element_bytes'];shape=spec['shape'];stride=spec['stride']
                unit=math.gcd(unit,spec['address']-self.storage[storage]['address'])
                for size,step in zip(shape[:-1],stride[:-1]):
                    if size>1:unit=math.gcd(unit,abs(step)*element)
                if shape:
                    span=((shape[-1]-1)*abs(stride[-1])+1)*element if shape[-1] else 0
                    unit=math.gcd(unit,span)
                    if stride[-1]!=1:unit=math.gcd(unit,element)
                else:unit=math.gcd(unit,element)
                self.dependency_granules[storage]=unit

    def dependency_granule(self,spec):
        if not hasattr(self,'dependency_granules'):self.initialize_dependency_granules()
        storage=spec['storage'] if isinstance(spec,dict) else spec
        return self.dependency_granules.get(storage,self.granule)

    def inheritance_masks(self,storage,tokens):
        """Physical sectors and byte-write strobes for fine logical tokens.

        Copy-forward preserves newly produced bytes sharing the same sector;
        these masks describe selective stores, never whole-sector memcpy.
        """
        unit=self.dependency_granule(storage)
        sectors,inverse=np.unique(tokens//(SECTOR_BYTES//unit),return_inverse=True)
        shift=(tokens%(SECTOR_BYTES//unit)*unit).astype(np.uint64)
        masks=np.zeros(len(sectors),np.uint64)
        np.bitwise_or.at(masks,inverse,np.left_shift(np.uint64((1<<unit)-1),shift))
        return sectors,masks

    def write_masks(self,spec,selected):
        """Byte enables for actual stores; a transfer sector is not a value."""
        starts=self.address_starts(spec,selected)
        width=(spec['shape'][-1] if spec['shape'] else 1)*spec['element_bytes']
        if spec['shape'] and spec['stride'][-1]!=1:
            columns=1 if spec['stride'][-1]==0 else spec['shape'][-1]
            starts=(starts[:,None]+np.arange(columns)*spec['stride'][-1]*spec['element_bytes']).ravel()
            width=spec['element_bytes']
        if not len(starts) or not width:return np.empty(0,np.int64),np.empty(0,np.uint64)
        if width%SECTOR_BYTES==0 and np.all(starts%SECTOR_BYTES==0):
            sectors=self.sectors(spec,selected)
            return sectors,np.full(len(sectors),(1<<SECTOR_BYTES)-1,np.uint64)
        offsets=np.arange(math.ceil((width+SECTOR_BYTES-1)/SECTOR_BYTES))
        sectors=starts[:,None]//SECTOR_BYTES+offsets
        begin=np.maximum(0,starts[:,None]-sectors*SECTOR_BYTES)
        end=np.minimum(SECTOR_BYTES,starts[:,None]+width-sectors*SECTOR_BYTES)
        valid=end>begin;tokens=sectors[valid]
        values=(np.left_shift(np.uint64(1),end[valid].astype(np.uint64))-
                np.left_shift(np.uint64(1),begin[valid].astype(np.uint64)))
        sectors,inverse=np.unique(tokens,return_inverse=True)
        masks=np.zeros(len(sectors),np.uint64);np.bitwise_or.at(masks,inverse,values)
        return sectors,masks

    def starts(self,spec,selected):
        shape=spec['shape']
        selected=np.asarray(selected,dtype=np.int64)
        if not shape:return np.zeros(len(selected),dtype=np.int64)
        offsets=np.zeros(len(selected),dtype=np.int64)
        remainder=selected.copy()
        for dim,stride in reversed(list(zip(shape[:-1],spec['stride'][:-1]))):
            offsets+=(remainder%dim)*stride
            remainder//=dim
        return (spec['address']-self.storage[spec['storage']]['address'])+offsets*spec['element_bytes']

    def sectors(self,spec,selected,granule=SECTOR_BYTES):
        return self.address_tokens(spec,selected,granule,physical=True)

    def address_starts(self,spec,selected):
        """Resolve physical row pitch separately from logical demand identity."""
        starts=self.starts(spec,selected)
        layout=getattr(self,'physical_views',{}).get(key(spec))
        if layout is None:return starts+self.bases[key(spec)]
        base,native_pitch,physical_pitch=layout
        row,column=np.divmod(starts,native_pitch)
        return base+row*physical_pitch+column

    def address_tokens(self,spec,selected,granule,physical):
        if len(selected)==0 or spec['bytes']==0:return np.empty(0,dtype=np.int64)
        starts=self.address_starts(spec,selected) if physical else self.starts(spec,selected)+self.bases[key(spec)]
        width=(spec['shape'][-1] if spec['shape'] else 1)*spec['element_bytes']
        if spec['shape'] and spec['stride'][-1]!=1:
            # Enumerate actual element locations of a transposed view.
            width=1 if spec['stride'][-1]==0 else spec['shape'][-1]
            points=starts[:,None]+np.arange(width)*spec['stride'][-1]*spec['element_bytes']
            # A logical token can be narrower than an aliased element.
            first=points//granule
            last=(points+spec['element_bytes']-1)//granule
            offsets=np.arange(math.ceil((spec['element_bytes']+granule-1)/granule))
            tokens=first[:,:,None]+offsets
            return np.unique(tokens[tokens<=last[:,:,None]])
        n=math.ceil((width+granule-1)/granule)
        ids=starts[:,None]//granule+np.arange(n)
        valid=ids*granule<starts[:,None]+width
        return np.unique(ids[valid])

    def mark(self,spec,selected=None):
        if spec['role']=='weight':return
        if selected is None:selected=np.arange(rows(spec))
        if len(selected)==0:return
        granule=self.dependency_granule(spec)
        identity=key(spec);mask=self.demand.get(identity)
        if mask is None:
            mask=np.zeros(math.ceil(self.storage[spec['storage']]['bytes']/granule),dtype=bool)
            self.demand[identity]=mask
        # A dense aligned row is a real contiguous run in this mask. Indexing
        # that row view avoids constructing/sorting a rows-by-width temporary
        # merely to set identical bits. Aliased/strided/partial rows retain
        # the general byte-token path below.
        dense=self.dense_demand_view(spec,mask,granule)
        if dense is not None and np.min(selected)>=0 and np.max(selected)<len(dense):
            dense[selected]=True
            return
        tokens=self.address_tokens(spec,selected,granule,physical=False)-self.bases[key(spec)]//granule
        if len(tokens) and (tokens.min()<0 or tokens.max()>=len(mask)):
            raise ValueError(('Out-of-storage dependency',spec,tokens.min(),tokens.max(),len(mask)))
        mask[tokens]=True

    def dense_demand_view(self,spec,mask,granule):
        if not spec['contiguous'] or not spec['shape'] or spec['stride'][-1]!=1:return None
        width=spec['shape'][-1]*spec['element_bytes']
        begin=spec['address']-self.storage[spec['storage']]['address']
        if width%granule or begin<0 or begin%granule:return None
        count=rows(spec);columns=width//granule
        stop=begin//granule+count*columns
        if not columns or stop>len(mask):return None
        return mask[begin//granule:stop].reshape(count,columns)

    def fully_written(self,spec):
        """Only whole granules suppress inheritance; partial bytes must carry."""
        if not spec['bytes']:return np.empty(0,dtype=np.int64)
        granule=self.dependency_granule(spec)
        if spec['contiguous']:
            begin=spec['address']-self.storage[spec['storage']]['address']
            end=begin+spec['bytes']
            stop=math.ceil(end/granule) if end==self.storage[spec['storage']]['bytes'] else end//granule
            return np.arange(math.ceil(begin/granule),stop,dtype=np.int64)
        if spec['shape'] and spec['stride'][-1]==1:
            starts=self.starts(spec,np.arange(rows(spec)))
            width=spec['shape'][-1]*spec['element_bytes']
            length=math.ceil((width+granule-1)/granule)
            tokens=starts[:,None]//granule+np.arange(length)
            valid=(tokens*granule>=starts[:,None])&((tokens+1)*granule<=starts[:,None]+width)
            return np.unique(tokens[valid])
        return np.empty(0,dtype=np.int64)

    def demanded(self,spec):
        if spec['bytes']==0:return np.zeros(rows(spec),dtype=bool)
        mask=self.demand.get(key(spec))
        if mask is None:return np.zeros(rows(spec),dtype=bool)
        granule=self.dependency_granule(spec)
        dense=self.dense_demand_view(spec,mask,granule)
        if dense is not None:return dense.any(1)
        starts=self.starts(spec,np.arange(rows(spec)))
        if spec['shape'] and spec['stride'][-1]!=1:
            width=1 if spec['stride'][-1]==0 else spec['shape'][-1]
            points=starts[:,None]+np.arange(width)*spec['stride'][-1]*spec['element_bytes']
            first=points//granule;last=(points+spec['element_bytes']-1)//granule
            offsets=np.arange(math.ceil((spec['element_bytes']+granule-1)/granule))
            tokens=first[:,:,None]+offsets
            return (mask[np.minimum(tokens,len(mask)-1)]&(tokens<=last[:,:,None])).any((1,2))
        width=(spec['shape'][-1] if spec['shape'] else 1)*spec['element_bytes']
        n=math.ceil((width+granule-1)/granule)
        ids=starts[:,None]//granule+np.arange(n)
        valid=ids*granule<starts[:,None]+width
        return (mask[np.minimum(ids,len(mask)-1)]&valid).any(1)

    def fully_written_rows(self,spec,selected):
        if spec['stride'][-1]!=1:
            raise NotImplementedError('Partial output writes require contiguous row elements')
        granule=self.dependency_granule(spec)
        starts=self.starts(spec,selected);width=spec['shape'][-1]*spec['element_bytes']
        tokens=starts[:,None]//granule+np.arange(math.ceil((width+granule-1)/granule))
        valid=(tokens*granule>=starts[:,None])&((tokens+1)*granule<=starts[:,None]+width)
        return np.unique(tokens[valid])

    def inputs_for(self,index,selected):
        event=self.events[index];kind=event['kind'];inputs=event['inputs'];outputs=event['outputs']
        result=[]
        if not inputs or not outputs:return result
        if event.get('op') in ('aten.new_full.default','aten.new_zeros.default','aten.new_ones.default',
                              'aten.full_like.default','aten.zeros_like.default','aten.ones_like.default'):
            # Tensor factories use the source's shape/dtype/device metadata,
            # never its element values. The output fill and real stores still
            # execute; inventing a source read also invents its entire closure.
            return result
        def add(s,ids=None):
            result.append((s,np.arange(rows(s),dtype=np.int64) if ids is None else np.asarray(ids,dtype=np.int64)))
        if 'diagnostic_control_rows' in event:
            for position,(s,mapping) in enumerate(zip(inputs,event['diagnostic_control_rows'])):
                if kind=='indexed_store' and position==0:continue
                add(s,np.asarray(mapping)[selected])
        elif event.get('control_primitive')=='scatter' and event.get('control_full_overwrite',False):
            # The captured integer permutation covers every destination
            # exactly once. The initial uninitialized destination is unused.
            for s in inputs[1:]:add(s)
        elif 'diagnostic_source_rows' in event:
            for s,mapping in zip(inputs[:-1],event['diagnostic_source_rows']):
                add(s,np.asarray(mapping)[selected])
            add(inputs[-1],selected)
        elif kind=='linear':
            if event.get('indexed_source',False):
                add(inputs[0],np.asarray(self.maps['indices'][index][0])[selected])
                for s in inputs[1:-1]:add(s)
                add(inputs[-1],selected)
            else:
                add(inputs[0],selected)
                for s in inputs[1:]:add(s)
        elif kind=='indexed_store':
            add(inputs[1],selected);add(inputs[2],selected)
        elif event.get('row_local_scalar_scatter',False):
            add(inputs[0],selected);add(inputs[1],selected)
        elif kind=='ancestor_merge':
            maps=self.maps['indices'][index]
            for j in range(5):add(inputs[j],np.asarray(maps[j])[selected])
            for j in range(5):add(inputs[5+j],selected)
            for base in (10,15):
                for j in range(5):add(inputs[base+j],np.asarray(maps[j])[selected])
            for s in inputs[20:]:add(s)
        elif kind=='progressive_center_distance':
            parents=np.asarray(self.maps['indices'][index][0])[selected]
            add(inputs[0],selected);add(inputs[1],parents);add(inputs[2],selected)
        elif kind=='progressive_difference':
            ids,parents=self.maps['indices'][index]
            add(inputs[0],selected);add(inputs[1],np.asarray(parents)[selected])
            add(inputs[2],np.asarray(ids)[selected]);add(inputs[3],selected)
        elif kind=='progressive_update':
            ids=np.asarray(self.maps['indices'][index][0])[selected]
            add(inputs[0],ids);add(inputs[1],selected);add(inputs[2],selected)
        elif kind=='progressive_observe':
            ids,parents=self.maps['indices'][index]
            add(inputs[0],selected);add(inputs[1],selected)
            for j in (2,3,4):add(inputs[j],np.asarray(ids)[selected])
            for j in (5,6,7):add(inputs[j])
            for j in (8,9):add(inputs[j],np.asarray(parents)[selected])
        elif kind=='spconv':
            mapping=np.asarray(self.maps['maps'][event['map_key']]['mapping'])
            ids=mapping[:,selected].ravel();add(inputs[0],np.unique(ids[ids>=0]))
            for s in inputs[1:]:add(s)
        elif kind=='ME':
            offsets=self.maps['maps'][event['map_key']]['offsets']
            active=np.zeros(rows(outputs[0]),dtype=bool);active[selected]=True
            ids=[np.asarray(pair[0])[active[np.asarray(pair[1])]] for pair in offsets]
            add(inputs[0],np.unique(np.concatenate(ids)))
            for s in inputs[1:]:add(s)
        elif kind=='sdpa' and len(inputs)>=3:
            shape=outputs[0]['shape'];q=shape[-2]
            bh=np.unique(selected//q)
            for j,s in enumerate(inputs):
                add(s,selected if j==0 else (bh[:,None]*s['shape'][-2]+np.arange(s['shape'][-2])).ravel())
        else:
            op=event.get('op','')
            source=inputs[0];out=outputs[0];nr=rows(out)
            if op=='aten.segment_reduce.default':
                from .segments import descriptor,source_rows
                offsets=descriptor(self,index)
                if offsets is not None:
                    add(source,source_rows(offsets,selected))
                    add(inputs[1],np.unique(np.concatenate((selected,selected+1))))
                    return result
            if op=='aten.mm.default':
                # An output row depends on that LHS row and the entire RHS.
                # In particular, a frozen semantic head is not a cross-row
                # reduction simply because it was captured as mm, not Linear.
                add(source,selected)
                for s in inputs[1:]:add(s)
                return result
            if (op in ('aten.abs.default','aten.square.default','aten.sqrt.default')
                and len(inputs)==1 and source['shape']==out['shape']):
                # These exact unary operators preserve element/row identity.
                # A small output tile must not reread every input row. Shape
                # equality alone never grants this rule to arbitrary ops.
                add(source,selected)
                return result
            if (op=='aten.mean.dim' and event.get('diagnostic_row_local',False)
                and len(inputs)==1 and len(source['shape'])==len(out['shape'])==2
                and source['shape'][0]==out['shape'][0] and out['shape'][1]==1):
                # The finite diagnostic compiler emits mean over channels,
                # keepdim=True. It is not a reduction across output rows.
                add(source,selected)
                return result
            from .reductions import row_local as reduction_row_local
            if reduction_row_local(event):
                add(source,selected)
                for s in inputs[1:]:add(s)
                return result
            if op.startswith('aten.copy_'):
                # Destination bytes are overwritten. Untouched storage ranges
                # are inherited explicitly by closure(), not read as operands.
                for s in inputs[1:]:add(s,selected if rows(s)==nr else None)
                return result
            if op=='aten.batch_norm.default' and event.get('batch_norm_training') is False and event.get('has_running_statistics'):
                add(source,selected)
                for s in inputs[1:]:add(s)
                return result
            index_values=self.maps['indices'].get(index)
            if op.startswith(('aten.index.Tensor','aten.index_select','aten.gather')) and index_values and len(source['shape'])==2 and len(out['shape'])==2:
                ids=np.asarray(index_values[0]).astype(np.int64)
                if len(ids)==nr and source['shape'][-1]==out['shape'][-1]:
                    resolved=ids[selected]
                    resolved=np.where(resolved<0,resolved+rows(source),resolved)
                    add(source,resolved)
                    for s in inputs[1:]:add(s,selected if s['role']=='index' and rows(s)==nr else None)
                    return result
            feature_inputs=[s for s in inputs if s['role']=='feature']
            rowcat=op.startswith('aten.cat') and len(out['shape'])==2 and feature_inputs and all(len(s['shape'])==2 and s['shape'][1]==out['shape'][1] for s in feature_inputs) and sum(s['shape'][0] for s in feature_inputs)==nr
            channelcat=(op.startswith('aten.cat') and len(out['shape'])==2 and feature_inputs
                and all(len(s['shape'])==2 and rows(s)==nr for s in feature_inputs)
                and sum(s['shape'][1] for s in feature_inputs)==out['shape'][1])
            base=0
            for s in inputs:
                if s['role']=='weight':
                    # A row-local dtype conversion reads these exact rows,
                    # even when its immutable source is a parameter. Matrix
                    # RHS and broadcast weights retain their full domain.
                    local_cast=(op.startswith('aten.to.') and len(s['shape'])==2 and
                                s['shape']==out['shape'])
                    add(s,selected if local_cast else None)
                    continue
                if rowcat and s['role']=='feature':
                    add(s,selected[(selected>=base)&(selected<base+rows(s))]-base);base+=rows(s)
                elif channelcat and s['role']=='feature':add(s,selected)
                elif rows(s)==nr and op.startswith(('aten.add.','aten.add_.','aten.sub.','aten.sub_.',
                    'aten.mul.','aten.div.','aten.relu','aten.gelu','aten.layer_norm',
                    'aten.clone','aten.copy_','aten.to.','aten.contiguous',
                    'aten.linalg_vector_norm','aten.clamp_min','aten.detach_',
                    'aten.ge.','aten.where.','aten.logical_','aten.bitwise_',
                    'aten.__and__','aten.__or__','aten.__rshift__','aten.__lshift__')):
                    add(s,selected)
                else:
                    add(s)
                    if s['role']=='feature' and len(selected)<nr:self.conservative[op or kind]+=1
        return result

    def closure(self,seeds,*,release_consumed=False):
        self.demand={};self.conservative=Counter()
        # Optional host-memory retirement for execution-only callers. Once
        # the reverse traversal passes the first producer of an SSA identity,
        # its demand has already propagated to every defining input. The live
        # rows and copy-forward ranges, rather than these scratch masks, own
        # the returned plan. Keep the diagnostic API's masks by default.
        first_producer={}
        if release_consumed:
            for i,event in enumerate(self.events):
                for spec in event['outputs']:first_producer.setdefault(key(spec),i)
        # Sparse map formation is implicit in opaque CUDA operators. All
        # captured integer/index construction remains mandatory, independently
        # of downstream feature pruning.
        for event in self.events:
            for spec in event['outputs']:
                if spec['role']=='index':self.mark(spec)
        for spec,selected in seeds:self.mark(spec,selected)
        live=[np.empty(0,dtype=np.int64) for _ in self.events]
        carry={}
        for i in range(len(self.events)-1,-1,-1):
            event=self.events[i]
            outputs=event['outputs']
            if not outputs:continue
            # A slice write increments the storage version but does not replace
            # untouched bytes. Carry their demand to the previous version.
            # Version-renamed physical buffers copy those demanded ranges.
            for identity in {key(s) for s in outputs}:
                mask=self.demand.get(identity)
                if mask is None or identity[1]==0:continue
                covered=np.zeros(len(mask),dtype=bool)
                for position,spec in enumerate(outputs):
                    if key(spec)==identity:
                        partial=(event['kind'] in ('progressive_update','indexed_store') and position==0) or (event['kind']=='progressive_observe' and position>=2)
                        tokens=(self.fully_written_rows(spec,np.asarray(self.maps['indices'][i][0]))
                                if partial else self.fully_written(spec))
                        covered[tokens]=True
                tokens=np.flatnonzero(mask&~covered)
                if len(tokens):
                    previous=(identity[0],identity[1]-1)
                    if previous not in self.bases:
                        # A fresh allocation has no initialized previous bytes.
                        # Padding inside a demand granule is not a dependency.
                        if identity[1]==1:continue
                        raise ValueError(('Missing inherited storage version',previous))
                    old=self.demand.setdefault(previous,np.zeros(len(mask),dtype=bool));old[tokens]=True
                    if (previous,identity) in self.inplace_versions:continue
                    if i in carry:raise NotImplementedError('Multiple partial output storages in one event')
                    carry[i]=(previous,identity,tokens)
            active=self.demanded(outputs[0])
            if event.get('control_primitive') in ('compact','unique'):
                # Empty compact/unique results still need their integer
                # program to establish the dynamic count/join completion.
                active[:]=True
            if event['kind'] in ('progressive_update','indexed_store'):
                touched=np.asarray(self.maps['indices'][i][0]);active=active[touched]
            if len(outputs)>1:
                from .reductions import row_local as reduction_row_local
                if reduction_row_local(event) and all(s['shape']==outputs[0]['shape'] for s in outputs):
                    # Values and indices/statistics share a row domain. A
                    # demand on the second output retains exactly those rows,
                    # including rows absent from the first output's demand.
                    for spec in outputs[1:]:active |= self.demanded(spec)
                elif any(self.demanded(s).any() for s in outputs[1:]):active[:]=True
            selected=np.flatnonzero(active)
            live[i]=selected
            if len(selected):
                for spec,ids in self.inputs_for(i,selected):self.mark(spec,ids)
            if release_consumed:
                for spec in outputs:
                    identity=key(spec)
                    if first_producer[identity]==i:self.demand.pop(identity,None)
        return DemandPlan(live,carry)

    def checkpoint_seeds(self,stops,witness=None):
        seeds=[]
        for depth,cp in enumerate(self.maps['checkpoints']):
            if depth<self.maps.get('observation_begin',0):continue
            if (witness is None and cp['name']!='final' and
                    depth>=self.maps.get('observation_end',len(self.maps['checkpoints'])-1)):continue
            active=np.flatnonzero(stops>=depth) if witness is None else np.flatnonzero(witness)
            if not len(active):continue
            if cp['name']=='final':
                active=active if witness is not None else np.flatnonzero(stops==depth)
                seeds.append((cp['spec'],np.unique(cp['mapping'][active])))
            else:
                # All observed readout operands are genuine output demands.
                for spec,mapping in zip(cp['readout_inputs'],cp['readout_maps']):
                    seeds.append((spec,np.unique(mapping[active])))
        return seeds

    def flops(self,index,selected=None):
        e=self.events[index]
        if not e['outputs']:return 0
        out=e['outputs'][0];count=rows(out) if selected is None else len(selected)
        if e['kind']=='linear':return 2*count*e['inputs'][0]['shape'][-1]*out['shape'][-1]
        if e['kind']=='ancestor_merge' and e['inline_fine']:
            return 2*count*e['channels'][0]*out['shape'][-1]
        if e['kind']=='spconv':
            m=np.asarray(self.maps['maps'][e['map_key']]['mapping'])
            edges=(m>=0).sum() if selected is None else (m[:,selected]>=0).sum()
            return int(2*edges*e['inputs'][0]['shape'][-1]*out['shape'][-1])
        if e['kind']=='ME':
            active=np.ones(rows(out),bool) if selected is None else np.isin(np.arange(rows(out)),selected)
            edges=sum(int(active[np.asarray(p[1])].sum()) for p in self.maps['maps'][e['map_key']]['offsets'])
            return 2*edges*e['inputs'][0]['shape'][-1]*out['shape'][-1]
        if e['kind']=='sdpa':return 4*count*out['shape'][-2]*out['shape'][-1]
        if e['kind']=='attention':
            lengths=np.diff(e['boundaries'])
            channels=e['inputs'][0]['shape'][-1]*e['inputs'][0]['shape'][-2]
            if selected is not None and len(selected)<rows(out):
                raise NotImplementedError('Native Flash partial-row arithmetic requires patch-level mapping')
            return int(4*(lengths**2).sum()*channels)
        if e.get('op')=='aten.mm.default':
            return 2*count*e['inputs'][0]['shape'][-1]*out['shape'][-1]
        return 0
