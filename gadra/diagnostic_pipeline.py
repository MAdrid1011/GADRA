"""Lower A observations onto the existing finite matrix/vector/memory engine.

Captured semantic/control observations remain the quality reference. This
module creates their shaped micro-operations, not another timing formula or
another compute engine. Diagnostic weights are cold, separately addressed
assets until their invocation-local composed projections become resident.
"""
from collections import Counter
import math
import numpy as np

from .diagnostics import equivalent_rows
from .graph import Graph,key,rows
from .predictor import uses_witness_selector
from .protocol import (SLOT_BYTES,SECTOR_BYTES,DMA_BURST_BYTES,DIAGNOSTIC_FORK_DESCRIPTOR_BYTES,
                       DIAGNOSTIC_WINDOW_DESCRIPTOR_BYTES,DIAGNOSTIC_DECISION_DESCRIPTOR_BYTES)


def source_register_words(graph,config):
    """One bounded FP32 row, with an explicit inherited width for children."""
    if not config.get('diagnostic_register_source',False):return 0
    widths=[config.get('diagnostic_source_register_words',0)]
    widths.extend(cp['readout_shape'][0] for cp in graph.maps.get('checkpoints',[]))
    words=max(widths)
    if words<1:raise ValueError('Register diagnostic source needs a finite row width')
    return int(words)


def register_source(sim,event,selected,earliest):
    """Normalize actual mapped source rows without an SRAM staging copy.

    Two row contexts hold the original FP32 center/norm/pad computation.
    Their vector arithmetic, native source reads, index readiness and final
    byte-enabled stores still use the shared execution resources. A context
    remains owned until its result store has returned.
    """
    if 'diagnostic_source_rows' not in event:raise ValueError('Register source requires actual mapped inputs')
    channels=event['source_channels'];out=event['outputs'][0];m=sim.memory;c=sim.c
    if channels+1>sim.diagnostic_source_register_words:raise ValueError('Diagnostic source row exceeds registers')
    index=event['inputs'][-1];parts=event['inputs'][:-1]
    loaded=m.access(sim.g.sectors(index,selected),ready=earliest)
    m.clock=max(m.clock,loaded)
    # Directory fills are real and bounded; they may overlap the row engine.
    # The subsequent reads still probe, wait for data and consume bank ports.
    for spec,mapping in zip(parts,event['diagnostic_source_rows']):
        if spec['element_bytes'] not in (2,4):raise ValueError('Unsupported source register precision')
        m.access(sim.g.sectors(spec,np.asarray(mapping)[selected]),prefetch=True)
    completed=earliest;lanes=c['vector_lanes']
    for row in selected:
        owner=int(sim.diagnostic_source_ready.argmin())
        m.clock=max(m.clock,int(sim.diagnostic_source_ready[owner]))+c['issue_cycles']
        ready=max(earliest,m.clock);cast_ops=0
        for spec,mapping in zip(parts,event['diagnostic_source_rows']):
            native=np.asarray([mapping[row]],np.int64)
            ready=max(ready,m.access(sim.g.sectors(spec,native),ready=m.clock))
            if spec['element_bytes']==2:cast_ops+=spec['shape'][-1]
        # Same vector elementary work as the materialized source: mean,
        # subtract, square/sum, norm clamp, normalize and bias insertion.
        blocks=math.ceil(channels/lanes)
        cost=(5*blocks if event['source_centered'] else 3*blocks)+2
        cost+=math.ceil(cast_ops/lanes)
        done=sim.issue_vector(ready,cost)
        sim.stats['vector_cycles']+=cost
        stored=sim.store(out,np.asarray([row],np.int64),done)
        sim.diagnostic_source_ready[owner]=stored;completed=max(completed,stored)
        sim.stats['diagnostic_source_register_admission_cycles']+=c['issue_cycles']
        sim.stats['diagnostic_source_register_cast_elements']+=cast_ops
    sim.stats['diagnostic_source_register_rows']+=len(selected)
    sim.stats['diagnostic_source_register_bytes']=sim.diagnostic_source_register_storage_bytes
    sim.stats['diagnostic_source_register_internal_read_bytes']+=len(selected)*channels*4*(
        5 if event['source_centered'] else 3)
    return int(completed)


def unique_index(sim,event,selected,earliest):
    """Run paid native dedup/rank in the existing finite output SRAM."""
    from .feeder import inverse_row_scratch,unique_row_transfer,unique_row_scratch,presence_rank_transfer
    native=np.asarray(event['diagnostic_unique_native'],np.int64);n=event['diagnostic_unique_rows']
    out=event['outputs'][0];c=sim.c;m=sim.memory
    if not np.array_equal(selected,np.arange(rows(out))):
        raise ValueError('Unique metadata needs one complete finite program')
    direct=c.get('diagnostic_direct_unique',False)
    layout=(unique_row_scratch if direct else inverse_row_scratch)(len(native),n)
    begin=max(earliest,sim.cycle,sim.vector,int(sim.buffer_ready[sim.active_output_buffer]))
    if layout[-1]>sim.output_budget():
        from .integer_stream import unique,COUNTERS,RECORD_BYTES
        # Preserve the explicit capacity failure when even a single radix
        # counter table plus one key/scan packet cannot fit the owner.
        if sim.output_budget()<2*COUNTERS*np.dtype(np.int64).itemsize+2*RECORD_BYTES:
            raise ValueError('Native index construction exceeds output owner, including its finite fallback')
        ready=unique(sim,next(i for i,e in enumerate(sim.g.events) if e is event),begin)[0]
        if (c.get('diagnostic_reuse_full_native_banks',False) or
                c.get('diagnostic_reuse_equal_native_packets',False)):
            ready=sim.issue_vector(ready,2)
            sim.stats['vector_cycles']+=2
            sim.stats['diagnostic_native_source_guard_cycles']+=2
        sim.stats['diagnostic_unique_peak_bytes']=max(sim.stats['diagnostic_unique_peak_bytes'],
            sim.stats['integer_control_peak_scratch_bytes'])
        return ready
    # Rebase the existing scratch into this actual accumulator owner, outside
    # all operand slots. Other arrays/owners can keep running; this owner's
    # previous result must have returned before any scratch byte is reused.
    base=sim.accumulator_base(4)
    banks,calendar,lengths=m.ports(2)
    transfer=presence_rank_transfer if direct else unique_row_transfer
    # An already increasing row-ID packet needs no full native-domain
    # presence/rank construction. Probe only a packet fitting the existing
    # input lanes; the original vector owner retains it through the decision.
    # Failure pays the probe and then executes the complete old microprogram.
    probe=(c.get('diagnostic_sorted_unique',False) and 0<len(native)<=c['vector_lanes']
           and event['inputs'][0]['element_bytes']==4)
    checked=begin;probe_work=0
    if probe:
        probe_work=3*math.ceil(len(native)/c['vector_lanes'])+math.ceil(math.log2(len(native)))+1
        checked+=probe_work
        sim.stats['diagnostic_unique_probe_cycles']+=probe_work
    fast=probe and np.all(native[1:]>native[:-1]) and native[0]>=0 and native[-1]<n
    if fast:
        used=native.copy();inverse=np.arange(len(native),dtype=np.int64)
        size=len(native)*4;scratch=2*size
        if scratch>sim.output_budget():raise ValueError('Native row packet exceeds output owner')
        ready=sim.output_transfer(size,checked,write=True,offset=base)
        generation=math.ceil(len(native)/c['vector_lanes']);ready+=generation
        ready=sim.output_transfer(size,ready,write=True,offset=base+size)
        ready=sim.output_transfer(scratch,ready,offset=base)
        work=probe_work+generation
        sim.stats['diagnostic_unique_fast_programs']+=1
        sim.stats['diagnostic_unique_fast_rows']+=len(native)
    else:
        if layout[-1]>sim.output_budget():raise ValueError('Native index construction exceeds output owner')
        ready,work,used,inverse=transfer(native,n,layout,checked,c['vector_lanes'],
            m.state,banks,calendar,lengths,c['bank_bytes_per_cycle'],m.counts,base)
        work+=probe_work;scratch=layout[-1]
    expected,route=np.unique(native,return_inverse=True)
    if not np.array_equal(used,expected) or not np.array_equal(inverse,route):
        raise AssertionError('Timed native dedup changed compact row identity')
    if (c.get('diagnostic_reuse_full_native_banks',False) or
            c.get('diagnostic_reuse_equal_native_packets',False)):
        # The existing unique controller returns the compact row count. Test
        # count==native-domain then branch before a consumer uses that proof.
        ready+=2;work+=2
        sim.stats['diagnostic_native_source_guard_cycles']+=2
    ready=int(ready)
    if c.get('vector_ready_queue',False):
        if sim.vector_queue.reserve(begin,ready-begin,begin)!=ready:
            raise AssertionError('Native metadata program lost its vector owner')
    sim.vector=max(sim.vector,ready)
    if sim.activity is not None:sim.activity.add_vector(begin,ready)
    sim.stats['vector_cycles']+=int(work)
    sim.stats['diagnostic_unique_programs']+=1
    sim.stats['diagnostic_unique_cycles']+=ready-begin
    sim.stats['diagnostic_unique_peak_bytes']=max(sim.stats['diagnostic_unique_peak_bytes'],scratch)
    return ready


class Program:
    # One bounded controller descriptor: fork/result identity, join PC and
    # valid/dispatch state. It shares control SRAM, not a private task queue.
    fork_descriptor_bytes=DIAGNOSTIC_FORK_DESCRIPTOR_BYTES
    def __init__(self,sim,depth):
        self.sim=sim;self.depth=depth
        g=self.g=Graph.__new__(Graph)
        g.events=[];g.maps=dict(indices={},maps={});g.map_bases={}
        g.bases=dict(sim.g.bases);g.storage=dict(sim.g.storage)
        g.conservative=Counter();g.inplace_versions=set();g.demand={}
        if not hasattr(sim,'diagnostic_assets'):
            sim.diagnostic_assets={}
            sim.diagnostic_next_id=max(sim.g.storage,default=0)+1
        self.external=set(sim.g.bases)
        # Invocation-local common subexpressions, not another feature cache.
        # Their ordinary graph readers extend the original SRAM lifetime.
        self.full_native_sources={}

    def tensor(self,shape,element=4,role='feature',asset=None):
        asset_key=(-1 if asset=='trajectory_state' else self.depth,asset) if asset is not None else None
        if asset_key is not None and asset_key in self.sim.diagnostic_assets:
            s=self.sim.diagnostic_assets[asset_key]
            if s['shape']!=list(shape) or s['element_bytes']!=element:raise ValueError('Diagnostic asset changed shape')
        else:
            identity=self.sim.diagnostic_next_id
            self.sim.diagnostic_next_id+=1
            size=math.prod(shape)*element
            address=self.sim.allocate_workspace(size)
            stride=[1]*len(shape)
            for i in range(len(shape)-2,-1,-1):stride[i]=stride[i+1]*shape[i+1]
            s=dict(storage=identity,version=0,address=address,shape=list(shape),stride=stride,
                   bytes=size,element_bytes=element,role=role,contiguous=True)
            if asset_key is not None:self.sim.diagnostic_assets[asset_key]=s
        self.g.storage[s['storage']]=dict(address=s['address'],bytes=s['bytes'])
        self.g.bases[key(s)]=s['address']
        if asset is not None:self.external.add(key(s))
        return s

    def refer(self,s):
        """Import a retained diagnostic value without inventing a new owner."""
        self.g.storage[s['storage']]=dict(address=s['address'],bytes=s['bytes'])
        self.g.bases[key(s)]=s['address'];self.external.add(key(s))
        return s

    def event(self,inputs,out,op,kind='aten',**extra):
        at=len(self.g.events)
        self.g.events.append(dict(kind=kind,op=op,stage='A.diagnostic',inputs=inputs,outputs=[out],diagnostic_row_local=True,**extra))
        return at

    def unary(self,source,element=None,ops=None,op='aten.mul.Tensor'):
        out=self.tensor(source['shape'],element or source['element_bytes'])
        extra={} if ops is None else dict(scalar_ops_per_row=ops)
        self.event([source],out,op,**extra)
        return out

    def gather(self,source,indices):
        indices=np.asarray(indices,np.int64)
        index=self.tensor((len(indices),1),4,'index')
        # Original mapping words arrive in the charged checkpoint prelude;
        # constructing a compact gather list is an actual finite vector/store
        # operation before its consumer can read it.
        self.event([],index,'aten.full.default',scalar_ops_per_row=2)
        out=self.tensor((len(indices),source['shape'][-1]),source['element_bytes'])
        at=self.event([source,index],out,'aten.index.Tensor',row_selector=True)
        self.g.maps['indices'][at]=[indices]
        return out

    def concat(self,parts):
        if len(parts)==1:return parts[0]
        out=self.tensor((rows(parts[0]),sum(s['shape'][-1] for s in parts)))
        self.event(parts,out,'aten.cat.default')
        return out

    def pad(self,source,width):
        if source['shape'][-1]==width:return source
        if source['shape'][-1]+1!=width:raise ValueError('Unknown readout padding')
        ones=self.tensor((rows(source),1));self.event([],ones,'aten.full.default')
        return self.concat([source,ones])

    def center(self,source,*,return_mean=False):
        mean=self.tensor((rows(source),1))
        self.event([source],mean,'aten.mean.dim',scalar_ops_per_row=source['shape'][-1])
        broadcast=dict(mean,shape=list(source['shape']),stride=[1,0],contiguous=False)
        out=self.tensor(source['shape'])
        self.event([source,broadcast],out,'aten.sub.Tensor')
        return (out,mean) if return_mean else out

    def normalize(self,source):
        norm=self.tensor((rows(source),1))
        self.event([source],norm,'aten.linalg_vector_norm.default',scalar_ops_per_row=2*source['shape'][-1])
        norm=self.unary(norm,op='aten.clamp_min.default')
        broadcast=dict(norm,shape=list(source['shape']),stride=[1,0],contiguous=False)
        out=self.tensor(source['shape'])
        self.event([source,broadcast],out,'aten.div.Tensor')
        return out

    def affine_source(self,parts,centered,width):
        """Stage one bounded row window, preserving center/norm/pad order.

        The execution engine charges its scratch passes in the existing
        output owner. There is no extra diagnostic cache or arithmetic unit.
        """
        channels=sum(s['shape'][-1] for s in parts)
        if width!=channels+1:raise ValueError('Affine source needs one bias column')
        out=self.tensor((rows(parts[0]),width))
        self.event(parts,out,'aten.mul.Tensor',diagnostic_source_fused=True,
                   source_channels=channels,source_centered=bool(centered))
        return out

    def mapped_affine_source(self,cp,representatives,recipe):
        """Read native checkpoint rows into the same finite FP32 scratch.

        One compact index payload is produced before any mapped source read.
        Maps are host lowering of that payload, not extra resident features.
        """
        parts=[cp['readout_inputs'][i] for i in recipe['input_order']]
        indices=[np.asarray(cp['readout_maps'][i])[representatives]
                 for i in recipe['input_order']]
        index=self.tensor((len(representatives),len(parts)),4,'index')
        self.event([],index,'aten.full.default',scalar_ops_per_row=2*len(parts))
        out=self.tensor((len(representatives),cp['readout_shape'][0]))
        self.event(parts+[index],out,'aten.mul.Tensor',diagnostic_source_fused=True,
                   diagnostic_source_rows=indices,source_channels=cp['readout_shape'][0]-1,
                   source_centered=bool(recipe['centered']))
        return out

    def linear(self,source,weight,bias=None):
        # Linear stores [N,K], whereas readout composition stores [K,N].
        w=dict(weight,shape=weight['shape'][::-1],stride=weight['stride'][::-1],contiguous=False)
        out=self.tensor((rows(source),weight['shape'][-1]),source['element_bytes'])
        self.event([source,w]+([] if bias is None else [bias]),out,'',kind='linear')
        return out

    def indexed_linear(self,source,weight,indices):
        """Feed a native row map directly into the existing matrix feeder."""
        indices=np.asarray(indices,np.int64)
        index=self.tensor((len(indices),1),4,'index')
        self.event([],index,'aten.full.default',scalar_ops_per_row=2)
        w=dict(weight,shape=weight['shape'][::-1],stride=weight['stride'][::-1],contiguous=False)
        out=self.tensor((len(indices),weight['shape'][-1]),source['element_bytes'])
        at=self.event([source,w,index],out,'',kind='linear',indexed_source=True)
        self.g.maps['indices'][at]=[indices]
        return out

    def unique_mapping(self,native,n):
        """Paid counting/rank program; its two aliases share one owner.

        The host unique is only the functional result of the finite program
        executed by unique_index. Both the gather list and inverse addresses
        are physically stored before their consumers can issue.
        """
        native=np.asarray(native,np.int64)
        used,inverse=np.unique(native,return_inverse=True)
        original=self.tensor((len(native),1),4,'index')
        self.event([],original,'aten.full.default',scalar_ops_per_row=2)
        payload=self.tensor((len(used)+len(native),1),4,'index')
        at=self.event([original],payload,'gadra.unique_index',
                      diagnostic_unique_native=native,diagnostic_unique_rows=n)
        self.g.events[at]['diagnostic_row_local']=False
        selected=dict(payload,shape=[len(used),1],bytes=len(used)*4)
        inverse_spec=dict(payload,address=payload['address']+len(used)*4,
                          shape=[len(native),1],bytes=len(native)*4)
        return used,inverse,selected,inverse_spec

    def partitioned_query(self,cp,representatives,folded,*,bias_source=None,output=False):
        """Native bank projections followed by paid mapped FP32 merging."""
        recipe=cp['readout_recipe']
        if folded['element_bytes']!=2:
            raise NotImplementedError('Partitioned query requires FP16 banks')
        centered=recipe['centered']
        composed=None
        if centered:
            composed=self.refer(bias_source if bias_source is not None else
                                self.sim.diagnostic_assets[(self.depth,'composed')])
        classes=folded['shape'][1];parts=[];maps=[];indices=[];begin=0;join=None;flops=0;centered_banks=[]
        for part in recipe['input_order']:
            spec=cp['readout_inputs'][part]
            native=np.asarray(cp['readout_maps'][part])[representatives]
            used,inverse,selected,inverse_spec=self.unique_mapping(native,rows(spec))
            unique_at=len(self.g.events)-1
            indices.append(inverse_spec)
            width=spec['shape'][-1]
            signature=(key(spec),spec['address'],tuple(spec['shape']),tuple(spec['stride']),
                       spec['element_bytes'],bool(centered))
            full=(self.sim.c.get('diagnostic_reuse_full_native_banks',False)
                  and len(used)==rows(spec))
            exact=self.sim.c.get('diagnostic_reuse_equal_native_packets',False)
            cached=self.full_native_sources.get(signature) if full or exact else None
            full_proof=full and cached is not None and len(cached[0])==rows(spec)
            if cached is not None and not full_proof:
                previous,packet,*_=cached
                if not exact or len(previous)!=len(used):cached=None
                else:
                    # Both sorted lists are already materialized. The shared
                    # integer/vector path compares every ID and reduces AND;
                    # equal cardinality alone never proves packet identity.
                    proof=self.tensor((1,1),4,'index')
                    unique_at=self.event([packet,selected],proof,'aten.equal.default',
                        scalar_ops_per_row=2*len(used),
                        scalar_dependency_cycles=math.ceil(math.log2(max(1,len(used))))+1)
                    self.g.events[unique_at]['diagnostic_row_local']=False
                    self.g.maps.setdefault('diagnostic_control_barriers',{})[unique_at]=dict(
                        cycles=0,read_bytes=0,producers=[unique_at])
                    self.sim.stats['diagnostic_native_packet_comparisons']+=1
                    if not np.array_equal(previous,used):cached=None
            if cached is None:
                source=self.tensor((len(used),width),spec['element_bytes'])
                at=self.event([spec,selected],source,'aten.index.Tensor',row_selector=True)
                self.g.maps['indices'][at]=[used]
                mean=None
                if centered:
                    if source['element_bytes']!=4:source=self.unary(source,element=4,op='aten.to.dtype')
                    source,mean=self.center(source,return_mean=True)
                norm=self.tensor((len(used),1))
                self.event([source],norm,'aten.mul.Tensor',scalar_ops_per_row=width*(2+int(source['element_bytes']==2)))
                half=source if source['element_bytes']==2 else self.unary(source,element=2,op='aten.to.dtype')
                if full or exact:self.full_native_sources[signature]=(used,selected,half,norm,mean)
            else:
                # Full valid domains or the paid ID comparison prove identical
                # native row order, without speculative parent suppliers.
                _,_,half,norm,mean=cached
                # An async suffix may restore an earlier dispatch frontier.
                # Explicitly wait for this unique's count/branch proof before
                # issuing any consumer of the reused preparation.
                self.g.maps.setdefault('diagnostic_control_barriers',{})[unique_at]=dict(
                    cycles=0,read_bytes=0,producers=[unique_at])
                self.sim.stats['diagnostic_native_source_reused_rows']+=len(used)
                self.sim.stats['diagnostic_native_source_reused_bytes']+=len(used)*width*spec['element_bytes']
            if centered:
                original=dict(composed,address=composed['address']+begin*composed['stride'][0]*4,
                              shape=[width,classes],bytes=width*classes*4,contiguous=False)
                centered_banks.append((mean,original,inverse))
            weight=dict(folded,address=folded['address']+begin*folded['stride'][0]*2,
                        shape=[width,classes],bytes=width*classes*2,contiguous=False)
            partial=self.linear(half,weight)
            if join is None:join=len(self.g.events)-1
            # Keep the native FP16 result in its existing owner. The merge
            # widens each consumed value before the original FP32 addition;
            # a separate full native-row FP32 copy is unnecessary. Repeated
            # consumers pay their own casts, not a free broadcast conversion.
            parts.extend([partial,norm]);maps.extend([inverse,inverse]);begin+=width
            flops+=2*len(used)*width*classes
        # Coefficients depend on the composed FP32 weight. Keep this branch
        # after the first product join; its reduction cannot be hoisted as
        # independent source work while cold composition is still running.
        for mean,original,inverse in centered_banks:
            asset=('centered_coefficient',original['storage'],original['version'],
                   original['address'],tuple(original['shape']),tuple(original['stride']))
            retained=self.sim.c.get('diagnostic_retain_center_coefficients',False)
            if retained and (self.depth,asset) in self.sim.diagnostic_assets:
                coefficient=self.refer(self.sim.diagnostic_assets[(self.depth,asset)])
                self.sim.stats['diagnostic_center_coefficient_reuses']+=1
            else:
                coefficient=self.tensor((classes,1),asset=asset if retained else None)
                self.event([self.transpose(original)],coefficient,'aten.sum.dim_IntList',
                           scalar_ops_per_row=original['shape'][0])
                self.sim.stats['diagnostic_center_coefficient_reductions']+=1
            parts.extend([mean,self.transpose(coefficient)])
            maps.extend([inverse,np.zeros(len(representatives),np.int64)])
        if composed is None:
            composed=self.refer(bias_source if bias_source is not None else
                                self.sim.diagnostic_assets[(self.depth,'composed')])
        bias=dict(composed,address=composed['address']+begin*composed['stride'][0]*4,
                  shape=[1,classes],bytes=classes*4,contiguous=False)
        parts.append(bias);maps.append(np.zeros(len(representatives),np.int64))
        parts.extend(indices);maps.extend([np.arange(len(representatives)) for _ in indices])
        query=self.tensor((len(representatives),classes))
        count=len(recipe['input_order'])
        from .diagnostics import affine_merge_cost
        operations,dependency=affine_merge_cost(classes,count,centered,output)
        self.event(parts,query,'aten.mul.Tensor',diagnostic_control_rows=maps,
                   scalar_ops_per_row=operations,
                   scalar_dependency_cycles=dependency,
                   diagnostic_centered_partition_widths=([cp['readout_inputs'][i]['shape'][-1]
                       for i in recipe['input_order']] if centered else []),
                   diagnostic_widen_partials=True,diagnostic_partitioned_output=output)
        if centered:self.sim.stats['diagnostic_centered_partitioned_rows']+=len(representatives)
        self.sim.stats['diagnostic_partition_cast_elements']+=len(representatives)*classes*count
        self.sim.stats['diagnostic_partitioned_output_rows' if output else 'diagnostic_partitioned_query_rows']+=len(representatives)
        self.sim.stats['diagnostic_partitioned_output_native_rows' if output else 'diagnostic_partitioned_native_rows']+=sum(len(np.unique(
            np.asarray(cp['readout_maps'][i])[representatives])) for i in recipe['input_order'])
        self.sim.stats['diagnostic_output_fp16_flops' if output else 'diagnostic_query_fp16_flops']+=flops
        return query,join

    def partitioned_language(self,cp,representatives,weight):
        """Reuse native bank products without staging a concatenated source.

        Every partial, norm and inverse uses the existing finite owners. The
        FP32 bias is taken from the original frozen language weight, not the
        rounded FP16 matrix. Only actually exiting representatives run this.
        """
        pack=self.sim.c.get('diagnostic_pack_language_weight',False)
        asset=(self.depth,'packed_language_half')
        if pack and asset in self.sim.diagnostic_assets:
            half=self.refer(self.sim.diagnostic_assets[asset]);self.external.add(key(half))
        else:
            half=self.query_cast(weight)
            if pack:
                half=self.pack_language_weight(half)
                self.sim.diagnostic_assets[asset]=half;self.external.add(key(half))
        value,_=self.partitioned_query(cp,representatives,half,bias_source=weight,output=True)
        return self.unary(value,ops=3*value['shape'][-1])

    @staticmethod
    def transpose(source):
        """A two-dimensional alias; no movement or matrix output scatter."""
        if len(source['shape'])!=2:raise ValueError('Expected a diagnostic matrix')
        return dict(source,shape=source['shape'][::-1],stride=source['stride'][::-1],contiguous=False)

    def text(self,width,classes):
        if self.sim.c.get('diagnostic_transposed_composition',False):
            # Captured texts physically store one language vector per class.
            return self.transpose(self.tensor((classes,width),role='weight',asset='text'))
        return self.tensor((width,classes),role='weight',asset='text')

    def query_cast(self,source):
        if (self.sim.c.get('diagnostic_transposed_composition',False) and
                source['stride']==[1,source['shape'][0]]):
            # Cast the physical dense rows, then retain the logical alias.
            # Casting logical columns into row-major storage would undo the
            # composition's producer/consumer layout agreement.
            return self.transpose(self.unary(self.transpose(source),element=2,op='aten.to.dtype'))
        return self.unary(source,element=2,op='aten.to.dtype')

    def compose_query(self,weight,text):
        if self.sim.c.get('diagnostic_transposed_composition',False):
            # (W @ T.T).T = T @ W.T. The matrix engine still writes ordinary
            # row-major outputs; no free transpose/scatter or extra SRAM.
            return self.transpose(self.linear(self.transpose(text),self.transpose(weight)))
        return self.linear(weight,text)

    def pack_language_weight(self,weight):
        """Pay for a dense physical RHS, retaining the logical affine map."""
        # Produced temporaries use the feature store, as do composed queries.
        # Declaring this a cold weight would send the feeder to another,
        # incoherent directory and miss the still-dirty producer's value.
        physical=self.tensor(weight['shape'][::-1],weight['element_bytes'])
        self.event([self.transpose(weight)],physical,'aten.clone.default')
        self.sim.stats['diagnostic_language_weight_pack_bytes']+=physical['bytes']
        return self.transpose(physical)

    def overlap_composition(self,composition,join):
        """Hoist only a cold weight composition and optional half cast.

        The source branch keeps its original topological order. Both branches
        issue through the existing finite owners; the query is their join.
        """
        events=self.g.events
        fork=[composition]
        if composition+1<join and events[composition+1].get('op')=='aten.to.dtype':
            cast=events[composition+1]
            if (len(cast['inputs'])==1 and
                key(cast['inputs'][0])==key(events[composition]['outputs'][0])):
                fork.append(composition+1)
        if events[composition]['kind']!='linear' or events[join]['kind']!='linear':
            raise ValueError('Diagnostic fork requires composition and query matrices')
        result=key(events[fork[-1]]['outputs'][0])
        if result not in {key(s) for s in events[join]['inputs']}:
            raise ValueError('Diagnostic query must join its composed weight')
        source=[i for i in range(join) if i not in fork]
        # Independent source microcode must not consume the composed result
        # or overwrite its operands, including another version of an alias.
        protected={s['storage'] for i in fork for s in events[i]['inputs']+events[i]['outputs']}
        fork_outputs={s['storage'] for i in fork for s in events[i]['outputs']}
        for i in source:
            e=events[i]
            independent_source=(e.get('diagnostic_row_local') or 'diagnostic_unique_native' in e)
            if (not independent_source or
                any(s['storage'] in fork_outputs for s in e['inputs']) or
                any(s['storage'] in protected for s in e['outputs'])):
                raise ValueError('Diagnostic source is not independent of composition')
        order=fork+source+list(range(join,len(events)))
        self.g.events=[events[i] for i in order]
        self.g.maps['indices']={new:self.g.maps['indices'][old]
            for new,old in enumerate(order) if old in self.g.maps['indices']}
        if self.g.maps.get('diagnostic_control_barriers'):
            remap={old:new for new,old in enumerate(order)}
            self.g.maps['diagnostic_control_barriers']={remap[at]:dict(barrier,
                producers=[remap[producer] for producer in barrier.get('producers',[])])
                for at,barrier in self.g.maps['diagnostic_control_barriers'].items()}
        self.g.maps['diagnostic_fork_join']=join

    def run(self,roots):
        g=self.g;g.initialize_dependency_granules()
        # Reuse physical units, owners and memory state. There is no private
        # diagnostic SRAM, DMA queue, array or free staging buffer.
        parent=self.sim
        final_output=key(roots[-1])==key(parent.g.events[-1]['outputs'][0])
        fork='diagnostic_fork_join' in g.maps
        window_descriptor=DIAGNOSTIC_WINDOW_DESCRIPTOR_BYTES if parent.c.get('diagnostic_row_windows',False) else 0
        decision_descriptor=DIAGNOSTIC_DECISION_DESCRIPTOR_BYTES if g.maps.get('diagnostic_control_barriers') else 0
        descriptor=(self.fork_descriptor_bytes if fork else 0)+window_descriptor+decision_descriptor
        c=dict(parent.c,control_sram_bytes=parent.c['control_sram_bytes']-descriptor,
               diagnostic_source_register_words=parent.diagnostic_source_register_words,
               stream_final_gather=final_output and parent.c.get('diagnostic_stream_multicast',False),stream_final_output=final_output and parent.c.get('stream_final_output',False),
               row_chain_max_matrices=0,row_chain_overlap=True)
        # This child has no coverage order or B controller.  Do not leak the
        # parent's constructor-time private-row flag into its A-only program;
        # it would fragment rows without performing any coverage scheduling.
        c['coverage_private_consumer_bypass_active']=False
        if parent.c.get('diagnostic_window_overlap',False):c['event_local_completion']=True
        child=type(parent)(g,c)
        child.allocate_workspace=parent.allocate_workspace
        for field in ('memory','matrix','buffer_ready','feeder_ready','feeder_b_ready','input_latches','copy_ready','copy_descriptor_queue','copy_source_order','vector_queue','activity','diagnostic_source_ready'):
            setattr(child,field,getattr(parent,field))
        child.cycle=parent.cycle;child.vector=parent.vector
        child.diagnostic_cast_ready=parent.diagnostic_cast_ready
        if window_descriptor:child.stats['diagnostic_window_descriptor_peak_bytes']=window_descriptor
        if decision_descriptor:
            child.stats['diagnostic_decision_descriptor_peak_bytes']=decision_descriptor
            child.stats['control_sram_write_bytes']+=decision_descriptor
            cost=math.ceil(decision_descriptor/(4*c['cam_lookups_per_cycle']))
            child.memory.clock=max(child.memory.clock,child.cycle)+cost
            child.cycle=max(child.cycle,child.memory.clock)
        if fork:
            cost=math.ceil(self.fork_descriptor_bytes/(4*c['cam_lookups_per_cycle']))+c['cam_latency_cycles']
            child.memory.clock=max(child.memory.clock,child.cycle)+cost
            child.cycle=max(child.cycle,child.memory.clock)
            child.stats['diagnostic_fork_programs']+=1
            child.stats['diagnostic_fork_control_cycles']+=cost
            child.stats['diagnostic_fork_descriptor_peak_bytes']=self.fork_descriptor_bytes
            child.stats['control_sram_write_bytes']+=self.fork_descriptor_bytes
        child.protected.update(self.external)
        child.protected.update(key(s) for s in roots)
        live=[e.get('dispatch_ids',np.arange(e.get('dispatch_rows',rows(e['outputs'][0])),dtype=np.int64)) for e in g.events]
        child.execute(live)
        for event in child.event_profile:
            parent.event_profile.append(dict(event,diagnostic_depth=self.depth))
        parent.cycle=max(child.cycle,parent.cycle);parent.vector=child.vector
        parent.diagnostic_cast_ready=child.diagnostic_cast_ready
        for name,value in child.stats.items():
            if (name.startswith('last_') or name.endswith('_peak_bytes') or name.endswith('_peak_elements') or name.endswith('_payload_bytes')
                or name in ('feeder_reserved_bytes','input_slice_latch_bytes','input_microprogram_bytes','operand_copy_register_bytes',
                            'direct_latch_reserved_bytes','direct_descriptor_bytes','direct_translation_reserved_bytes',
                            'attention_ready_queue_bytes','attention_softmax_register_bytes','attention_queue_peak',
                            'diagnostic_source_register_bytes',
                            'vector_ready_queue_bytes','vector_ready_queue_peak',
                            'input_microprogram_peak_bank_entries')):
                parent.stats[name]=max(parent.stats[name],value)
            else:parent.stats[name]+=value
        # Scratch has no future consumer after this checkpoint. Its bytes may
        # be discarded only after the full diagnostic chain has completed.
        for e in g.events:
            for s in e['outputs']:
                if key(s) in self.external or key(s) in {key(r) for r in roots}:continue
                sectors=g.sectors(s,np.arange(rows(s)))
                cost=math.ceil(len(np.unique(sectors//(SLOT_BYTES//SECTOR_BYTES)))/c['cam_lookups_per_cycle'])+c['cam_latency_cycles']
                parent.memory.clock=max(parent.memory.clock,parent.cycle)+cost
                parent.stats['diagnostic_retirement_cycles']+=cost
                parent.memory.discard(sectors,parent.cycle)
        parent.cycle=max(parent.cycle,parent.memory.clock)
        return child.stats


def readout(p,cp,representatives):
    """Captured recipe, shared by control, output and completed feedback."""
    if cp.get('terminal_latent_query',False):
        if len(cp['readout_inputs'])!=1 or len(cp['readout_maps'])!=1:
            raise ValueError('Terminal latent query requires one native source map')
        source=p.gather(cp['readout_inputs'][0],
                        np.asarray(cp['readout_maps'][0])[representatives])
        source=p.pad(source,cp['readout_shape'][0])
        weight=p.tensor(tuple(cp['readout_shape']),role='weight',asset='language_weight')
        return source,weight,cp['readout_shape'][0]
    if 'readout_recipe' in cp:
        from .readout import validate_recipe
        recipe=validate_recipe(cp,getattr(p.sim.g,'trace',{}).get('model'))
        if p.sim.c.get('diagnostic_source_direct',False) or p.sim.c.get('diagnostic_register_source',False):
            source=p.mapped_affine_source(cp,representatives,recipe)
            weight=p.tensor(tuple(cp['readout_shape']),role='weight',asset='language_weight')
            return source,weight,cp['readout_shape'][0]
        parts=[p.gather(cp['readout_inputs'][i],np.asarray(cp['readout_maps'][i])[representatives])
               for i in recipe['input_order']]
        parts=[p.unary(s,element=4,op='aten.to.dtype') if s['element_bytes']!=4 else s for s in parts]
        if p.sim.c.get('diagnostic_source_fusion',False):
            source=p.affine_source(parts,recipe['centered'],cp['readout_shape'][0])
        else:
            source=p.concat(parts)
            if recipe['centered']:source=p.center(source)
            source=p.pad(p.normalize(source),cp['readout_shape'][0])
        weight=p.tensor(tuple(cp['readout_shape']),role='weight',asset='language_weight')
        return source,weight,cp['readout_shape'][0]
    k,width=cp['readout_shape'];lang=cp['semantic_width']
    native=bool(cp['readout_extra_flops_per_row']);identity=cp.get('readout_identity',False)
    inner=lang if identity else width+1 if native else k
    weight=None if identity else p.tensor((inner,lang),role='weight',asset='language_weight')
    if native and p.sim.c.get('diagnostic_indexed_native_readout',False):
        if len(cp['readout_inputs'])!=1 or len(cp['readout_maps'])!=1:
            raise ValueError('Indexed native readout requires one captured source map')
        kernel=p.tensor((k,width),role='weight',asset='latent_kernel')
        native_rows=np.asarray(cp['readout_maps'][0])[representatives]
        source=p.indexed_linear(cp['readout_inputs'][0],kernel,native_rows)
        return p.unary(source,op='aten.relu.default'),weight,inner
    parts=[p.gather(s,np.asarray(mapping)[representatives])
           for s,mapping in zip(cp['readout_inputs'],cp['readout_maps'])]
    if identity and not parts:parts=[p.gather(p.sim.g.events[-1]['outputs'][0],representatives)]
    parts=[p.unary(s,element=4,op='aten.to.dtype') if s['element_bytes']!=4 else s for s in parts]
    if not native and not identity:parts=[p.unary(s,ops=3*s['shape'][-1]) for s in parts]
    source=p.concat(parts)
    if not native and not identity:source=p.unary(source,ops=3*source['shape'][-1])
    source=p.pad(source,source['shape'][-1] if identity else k)
    if native:
        kernel=p.tensor((k,width),role='weight',asset='latent_kernel')
        source=p.unary(p.linear(source,kernel),op='aten.relu.default')
    return source,weight,inner


def language(p,source,weight,inner,output_fp16=False,source_rows=None):
    if output_fp16 and weight is not None:
        source=p.pad(source,inner)
        if source['element_bytes']!=2:
            source=p.unary(source,element=2,op='aten.to.dtype')
        weight=p.query_cast(weight)
    if weight is not None and p.sim.c.get('diagnostic_pack_language_weight',False):
        if p.sim.c.get('diagnostic_row_windows',False):
            # All windows at this checkpoint use this same frozen affine.
            # Keep its paid copy in the ordinary feature directory; a miss
            # still pays the normal reload and finite SRAM/bank costs.
            asset=(p.depth,'packed_language_half' if output_fp16 else 'packed_language')
            if asset not in p.sim.diagnostic_assets:
                p.sim.diagnostic_assets[asset]=p.pack_language_weight(weight)
            weight=p.refer(p.sim.diagnostic_assets[asset])
            p.external.add(key(weight))
        else:weight=p.pack_language_weight(weight)
    if weight is None:
        if source_rows is not None:raise ValueError('Indexed language output requires a matrix weight')
        result=source
    elif source_rows is not None:
        result=p.indexed_linear(source,weight,source_rows)
    else:result=p.linear(p.pad(source,inner),weight)
    if output_fp16 and result['element_bytes']!=4:result=p.unary(result,element=4,op='aten.to.dtype')
    return p.unary(result,ops=3*result['shape'][-1])


def prepare_query(p,cp,representatives,classes,windowed):
    """Shared paid current-query construction for observations and owners."""
    sim=p.sim;depth=p.depth;lang=cp['semantic_width']
    partitioned=cp.get('query_partitioned',False)
    if partitioned:
        latent=None;query_inner=cp['readout_shape'][0]
        language_weight=p.tensor(tuple(cp['readout_shape']),role='weight',asset='language_weight')
    else:latent,language_weight,query_inner=readout(p,cp,representatives)
    text=p.text(lang,classes)
    composition=None
    factors=cp.get('query_factor_shapes')
    query_source=latent
    if factors:
        if partitioned or language_weight is None or len(factors)!=2:
            raise ValueError('Factorized query requires one ordinary affine readout')
        left_shape,right_shape=map(tuple,factors)
        if (left_shape[0]!=query_inner or left_shape[1]!=right_shape[0]
                or right_shape[1]!=lang):
            raise ValueError('Captured query factors do not match the readout')
        left=p.tensor(left_shape,element=2,role='weight',asset='query_factor_left')
        right=p.tensor(right_shape,role='weight',asset='query_factor_right')
        factor_source=p.unary(p.pad(latent,left_shape[0]),element=2,op='aten.to.dtype')
        query_source=p.linear(factor_source,left)
        projection_inner=left_shape[1]
        language_query_weight=right
    else:
        projection_inner=query_inner
        language_query_weight=language_weight
    if language_query_weight is None:folded=text
    elif depth not in sim.prepared_readouts:
        # Composition is an invocation-local FP32 matrix operation, with
        # a real dependency before any folded projection can consume it.
        folded=p.compose_query(language_query_weight,text)
        composition=len(p.g.events)-1
        p.sim.diagnostic_assets[(depth,'composed')]=folded
        p.external.add(key(folded));sim.prepared_readouts.add(depth)
        sim.stats['query_projection_preparations']+=1
    else:
        folded=sim.diagnostic_assets[(depth,'composed')]
        p.g.storage[folded['storage']]=dict(address=folded['address'],bytes=folded['bytes'])
        p.g.bases[key(folded)]=folded['address'];p.external.add(key(folded))
    half=cp.get('query_fp16',False)
    if half:
        # Cast of the composed weights is retained for every observation;
        # both matrix precision and every cast have finite service time.
        half_asset=(depth,'composed_half')
        if windowed and half_asset in sim.diagnostic_assets:
            folded=p.refer(sim.diagnostic_assets[half_asset])
        else:
            folded=p.query_cast(folded)
            if windowed:
                sim.diagnostic_assets[half_asset]=folded
                p.external.add(key(folded))
    if partitioned:query,join=p.partitioned_query(cp,representatives,folded)
    else:
        query_source=p.pad(query_source,projection_inner)
        if half and query_source['element_bytes']!=2:
            query_source=p.unary(query_source,element=2,op='aten.to.dtype')
        query=p.linear(query_source,folded)
        join=len(p.g.events)-1
        if half:query=p.unary(query,element=4,op='aten.to.dtype')
    return query,join,composition,language_weight,latent,query_source,query_inner,partitioned,half


def retire(sim,spec):
    sectors=np.arange(spec['address']//SECTOR_BYTES,
                      math.ceil((spec['address']+spec['bytes'])/SECTOR_BYTES))
    cost=math.ceil(len(np.unique(sectors//(SLOT_BYTES//SECTOR_BYTES)))/sim.c['cam_lookups_per_cycle'])+sim.c['cam_latency_cycles']
    sim.memory.clock=max(sim.memory.clock,sim.cycle)+cost
    sim.stats['diagnostic_retirement_cycles']+=cost
    sim.memory.discard(sectors,sim.memory.clock)
    sim.cycle=max(sim.cycle,sim.memory.clock)


def release_language_pack(sim,depth):
    for asset in ('packed_language','packed_language_half'):
        spec=getattr(sim,'diagnostic_assets',{}).pop((depth,asset),None)
        if spec is not None:retire(sim,spec)


def trajectory_state_width(classes,config):
    # Only query-history lanes are packed. TV ring/sum/count and optional
    # winner/valid words retain their original FP32/int32 representation.
    query_words=math.ceil(classes/2) if config.get('diagnostic_history_fp16',False) else classes
    words=query_words+config['predictor_window']+1+2*int(config.get('response_match_winner',False))
    return math.ceil(words*4/SECTOR_BYTES)*SECTOR_BYTES//4


def row_window_size(sim,classes):
    """One fixed leaf-ID partition across depths, bounded by existing owners.

    A window must fit one widest FP32 row set in the current output owner and
    its descriptors in the existing issue queue. Physical temporaries still
    use the ordinary feature store; this is no additional scratch SRAM.
    """
    checkpoints=sim.g.maps['checkpoints']
    state=trajectory_state_width(classes,sim.c)
    if sim.c.get('diagnostic_query_windows',False):
        # Partitioned native-bank queries never materialize the wide latent.
        # Bound a control cohort by query, retained state and native mapping
        # words. The wide exit suffix uses the ordinary independently tiled
        # output path, not a whole-cohort latent in this owner. This is an
        # admission budget; actual temporaries still occupy the feature cache
        # and all extra projections/spills are charged by Program.run.
        observed=checkpoints[:-1]
        if not sim.c.get('diagnostic_direct_state',False) or not observed or not all(
                cp.get('query_partitioned',False) for cp in observed):
            raise ValueError('Query windows require partitioned native-bank queries and direct state')
        width=(classes+state+max(len(cp['readout_inputs']) for cp in checkpoints))*4
    else:
        widths=[classes,state]
        for cp in checkpoints:
            widths.extend([cp['semantic_width'],*cp['readout_shape'][:1]])
        width=max(widths)*4
    rows_=min(sim.output_budget()//width,sim.c['queue_entries']*sim.c['array_rows'])
    if rows_<1:raise ValueError('A diagnostic row does not fit the output owner')
    if rows_>=sim.c['array_rows']:rows_=rows_//sim.c['array_rows']*sim.c['array_rows']
    return rows_


def mappings(sim,checks):
    """Finite mapping DMA plus vector/index construction, before consumers."""
    loaded=sim.cycle
    for begin in range(0,checks*4,DMA_BURST_BYTES):
        returned=sim.memory.core.transaction(sim.cycle,False,sim.memory.state,sim.memory.pending,
            sim.memory.counts,sim.memory.burst_cycles,sim.c['dram_latency_cycles'],DMA_BURST_BYTES,sim.memory.dma_activity)
        payload=min(DMA_BURST_BYTES,checks*4-begin)
        loaded=max(loaded,int(returned))+math.ceil(payload/(4*sim.c['cam_lookups_per_cycle']))
    sim.cycle=loaded
    work=math.ceil(checks/sim.c['vector_lanes'])
    sim.cycle=sim.issue_vector(sim.cycle,work)
    sim.stats['vector_cycles']+=work
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['diagnostic_mapping_read_bytes']+=checks*4


def control_rounds(addresses,ports):
    """32-bit word addresses; one word per bank per cycle."""
    return int(np.bincount(np.asarray(addresses,dtype=np.int64).ravel()%ports,
                           minlength=ports).max(initial=0))


def feedback_gate_header_bytes(config):
    """Cold prior and per-axis support words in the existing control SRAM."""
    return 4*(2+int(config.get('predictor_joint_response',False)))


def feedback_gate_storage(context_rows,leaf_rows,config):
    """Two masks, support/walker words and one bounded selected-ID packet."""
    if not config.get('diagnostic_feedback_gate',False):return 0
    # Prior flag, response-bin support mask; PC, limit, valid and word cursor.
    if config['predictor_response_bins']>32:raise ValueError('Response support exceeds one 32-bit word')
    return feedback_gate_header_bytes(config)+4*(math.ceil(context_rows/32)+math.ceil(leaf_rows/32)+4+
              config['queue_entries']*config['array_rows'])


def initialize_feedback_gate(sim,witnesses):
    """Publish a necessary-condition mask after real pilot feedback commits.

    Context eligibility is reduced using completed owner records and frozen
    LUT support, then expanded through the actual leaf/context map. Nothing
    depends on the remaining leaves' scores, future labels or chosen stops.
    The banked control SRAM holds both bitmaps until invocation retirement.
    """
    from .predictor import feedback_gate_contexts,response_record_words
    if not getattr(sim,'feedback_published',False):raise ValueError('Gate reads unfinished feedback')
    if hasattr(sim,'feedback_gate_leaves'):raise ValueError('Gate is invocation-local and published once')
    if not hasattr(sim,'predictor_gate_support'):raise ValueError('Gate support was not compiled')
    c=sim.c;ports=c['cam_lookups_per_cycle'];depth=c['predictor_first_exit']
    keys,values,group=sim.completed_response_records
    tolerance=min(c['semantic_tolerance'],c['class_repair_tolerance'])
    if c.get('predictor_joint_response',False):
        from .predictor import joint_context_actions
        rb=c['predictor_response_bins']
        possible,_=joint_context_actions(keys[depth],values[depth],sim.predictor_response_edges[depth],
            sim.predictor_nearest_edges[depth],sim.predictor_response_actions[depth].reshape(-1,rb),
            sim.predictor_nearest_actions[depth].reshape(-1,rb),tolerance)
    else:
        possible=feedback_gate_contexts(keys[depth],values[depth],sim.predictor_response_edges[depth],
                                       sim.predictor_gate_support,tolerance)
    context=sim.g.maps['response_context'];mapping=np.asarray(context['mapping'])
    actual=np.unique(mapping);count=rows(context['spec']);slots=c.get('witnesses_per_context',1)
    flags=np.zeros(count,bool);flags[actual]=possible
    start=max(sim.cycle,sim.memory.clock,sim.feedback_completed_cycle)
    fields=response_record_words(c)
    header_bytes=feedback_gate_header_bytes(c)
    sim.cycle=start+math.ceil(header_bytes/(4*ports))+c['cam_latency_cycles']
    sim.stats['control_sram_read_bytes']+=header_bytes
    # Produce one packed bitmap word at a time, never an unbounded record
    # array. At most32 contexts' finite owner sets occupy existing input lanes;
    # narrow synthetic configurations subdivide them further. Clear the word,
    # read/evaluate each subpacket, then OR/write this word before proceeding.
    packet=min(32,c['vector_lanes']//(slots*fields))
    if packet<1:raise ValueError('A complete owner set must fit the existing vector input lanes')
    for first in range(0,count,32):
        local=actual[(actual>=first)&(actual<first+32)]
        sim.cycle+=1
        sim.stats['control_sram_write_bytes']+=4
        for offset in range(0,len(local),packet):
            batch=local[offset:offset+packet]
            address=(((batch[:,None]*len(sim.g.maps['checkpoints'])+depth)*slots+
                      np.arange(slots))[...,None]*fields+np.arange(fields))
            reads=control_rounds(address,ports)+c['cam_latency_cycles']
            work=len(batch)*(slots*slots*3+slots*(4+len(sim.predictor_response_edges[depth])))
            if c.get('packed_response_records',False):work+=2*len(batch)*slots
            if c.get('predictor_joint_response',False):
                # Two independent necessary action supports; real owner reads
                # above are shared, nearest bin comparisons/OR are additional.
                work+=len(batch)*slots*(4+len(sim.predictor_nearest_edges[depth]))
            cost=math.ceil((work+len(batch)+31)/c['vector_lanes'])+5
            sim.cycle=sim.issue_vector(sim.cycle+reads+1,cost)+1
            sim.stats['vector_cycles']+=cost
            sim.stats['control_sram_read_bytes']+=address.size*4+4
            sim.stats['control_sram_write_bytes']+=4
    # The map is real per-invocation data. No retained full leaf->context table
    # or preloaded vector of all context flags is assumed.
    batch_width=32*max(1,ports//32)
    for begin in range(0,len(mapping),batch_width):
        batch=mapping[begin:begin+batch_width]
        # Only one returned batch occupies the existing input lanes; the
        # next mapping packet is not requested until those lanes are released.
        mappings(sim,len(batch))
        # A bank word read may multicast inside this one finite lane batch.
        words=np.unique(batch//32)
        service=control_rounds(words,ports)+c['cam_latency_cycles']
        packed=math.ceil(len(batch)/32)
        # Word/bit address, extract and 32-lane OR tree to pack the leaf mask.
        arithmetic=math.ceil((3*len(batch)+31*packed)/c['vector_lanes'])+5
        sim.cycle=sim.issue_vector(max(sim.cycle,sim.memory.clock)+service,arithmetic)
        sim.stats['vector_cycles']+=arithmetic
        sim.cycle+=math.ceil(packed/ports)
        sim.stats['control_sram_read_bytes']+=len(words)*4
        sim.stats['control_sram_write_bytes']+=packed*4
    sim.feedback_gate_leaves=flags[mapping]
    sim.feedback_gate_ready=sim.cycle
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['feedback_gate_reserved_bytes']=feedback_gate_storage(count,len(mapping),c)
    sim.stats['feedback_gate_rejected_leaves']=int((~sim.feedback_gate_leaves&~witnesses).sum())
    elapsed=sim.cycle-start
    sim.stats['feedback_gate_setup_cycles']+=elapsed
    sim.stats['diagnostic_cycles']+=elapsed


def gated_observations(sim,selected,classes):
    """Bounded ordered mask walker feeding the existing diagnostic mapper.

    It scans immutable IDs into one reserved finite control-SRAM ID packet.
    The packet remains owned through this window's Program.run; it does not
    replace the mapper's paid native dedup or mapped input construction.
    """
    if not hasattr(sim,'feedback_gate_leaves'):raise ValueError('Observation gate is not published')
    if getattr(sim,'feedback_gate_packet_owned',False):raise ValueError('Previous gate packet is still live')
    limit=row_window_size(sim,classes)
    if len(selected)>limit or len(np.unique(selected//limit))>1:
        raise ValueError('Gate walker must be consumed within one diagnostic row window')
    c=sim.c;ports=c['cam_lookups_per_cycle'];start=max(sim.cycle,sim.memory.clock,sim.feedback_gate_ready)
    words=np.unique(selected//32)
    read=control_rounds(words,ports)+c['cam_latency_cycles']
    # Fetch PC/limit/valid/cursor, extract each predicate and advance the
    # ordered row walker through the original finite issue lanes.
    # Per issue batch, a prefix-rank tree routes valid immutable IDs into the
    # reserved packet. Output occupancy cannot exceed the original row limit.
    cost=math.ceil(3*len(selected)/c['vector_lanes'])
    cost+=math.ceil(len(selected)/ports)*math.ceil(math.log2(ports))
    ready=sim.issue_vector(start+math.ceil(4/ports)+read+math.ceil(4/ports),cost)
    sim.cycle=ready+math.ceil(len(selected)/ports)+math.ceil(4/ports)
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['vector_cycles']+=cost
    sim.stats['control_sram_read_bytes']+=len(words)*4+16
    sim.stats['control_sram_write_bytes']+=32
    result=selected[sim.feedback_gate_leaves[selected]]
    sim.cycle+=math.ceil(len(result)/ports)
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['control_sram_write_bytes']+=len(result)*4
    sim.stats['feedback_gate_packet_peak_bytes']=max(sim.stats['feedback_gate_packet_peak_bytes'],len(result)*4)
    sim.feedback_gate_packet_owned=bool(len(result))
    sim.stats['feedback_gate_skipped_observations']+=len(selected)-len(result)
    sim.stats['feedback_gate_walk_cycles']+=sim.cycle-start
    return result


def consume_gate_packet(sim,count):
    """The packet is read into the original mapper, retained until program end."""
    if not getattr(sim,'feedback_gate_packet_owned',False):raise ValueError('No valid gate ID packet')
    sim.cycle=max(sim.cycle,sim.memory.clock)+math.ceil(count/sim.c['cam_lookups_per_cycle'])+sim.c['cam_latency_cycles']
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['control_sram_read_bytes']+=count*4


def release_gate_packet(sim):
    """Clear valid only after mapped indices have their independent owners."""
    if not getattr(sim,'feedback_gate_packet_owned',False):raise ValueError('No valid gate ID packet')
    sim.cycle=max(sim.cycle,sim.memory.clock)+1
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['control_sram_write_bytes']+=4
    sim.feedback_gate_packet_owned=False


def select_witnesses(sim,witnesses):
    """Two finite scans after the actual coarsest mapping is produced.

    Counts/ranks/owners and the exclusion bitmap stay allocated throughout
    both phases. Up to eight compile-time owner ranks use fixed multiply-high
    or shift thresholds; the host sorting helper is not a free selector.
    """
    from .predictor import (witness_layout,selector_storage_bytes,
                            selector_owner_addresses,selector_owner_bytes)
    if hasattr(sim,'response_owner_slots'):raise ValueError('Pilot owners already selected')
    c=sim.c;slots=c.get('witnesses_per_context',1)
    if not 1<=slots<=8 or c['witness_stride']!=1:
        raise NotImplementedError('Finite owner selection supports one to eight slots in every context')
    context=sim.g.maps['response_context'];contexts=np.asarray(context['mapping'])
    count=rows(context['spec']);ports=c['cam_lookups_per_cycle'];start=sim.cycle
    _,owners=witness_layout(contexts,c)
    actual=np.full((count,slots),-1,np.int64);actual[np.unique(contexts)]=owners
    selected=np.sort(actual[actual>=0])
    if not np.array_equal(selected,np.flatnonzero(witnesses)):
        raise ValueError('Pilot selection disagrees with finite owners')
    size=selector_storage_bytes(count,len(contexts),c)
    # Clear count/rank/owner/bitmap words via their existing control ports.
    clear=math.ceil(size/(4*ports))
    sim.cycle=max(sim.cycle,sim.memory.clock)+clear
    sim.stats['control_sram_write_bytes']+=size
    mappings(sim,2*len(contexts))
    service=0
    for begin in range(0,len(contexts),ports):
        ctx=contexts[begin:begin+ports]
        # Count read/update/write, then count + rank read/update/write.
        # Repeated contexts within a batch serialize on the same bank.
        service+=5*control_rounds(ctx*2,ports)
    owner_addresses=selector_owner_addresses(np.repeat(np.arange(count),slots),
        np.tile(np.arange(slots),count),count,len(contexts),c)[actual.ravel()>=0]
    service+=control_rounds(owner_addresses,ports)
    # Exclusion bits require read/OR/write with same-word conflicts.
    service+=2*control_rounds(selected//32,ports)
    arithmetic=math.ceil((7*len(contexts)+len(selected))/c['vector_lanes'])
    if slots>2:
        # Form every fixed j*count/K rank and compare the running rank against
        # it. Power-of-two K uses shifts; other K uses constant multiply-high.
        # Short contexts still retain min(slots,count) distinct ranks.
        per_row=(2 if slots&(slots-1)==0 else 4)*(slots-1)+(slots-2)
        extra=math.ceil(per_row*len(contexts)/c['vector_lanes'])
        arithmetic+=extra
        sim.stats['witness_rank_integer_cycles']+=extra
    sim.cycle=sim.issue_vector(max(sim.cycle,sim.memory.clock),service+arithmetic)
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['vector_cycles']+=arithmetic
    owner_bytes=selector_owner_bytes(len(contexts))
    sim.stats['control_sram_read_bytes']+=12*len(contexts)+owner_bytes*len(selected)
    sim.stats['control_sram_write_bytes']+=8*len(contexts)+(4+owner_bytes)*len(selected)
    sim.stats['witness_selection_cycles']+=sim.cycle-start
    sim.stats['witness_selector_reserved_bytes']=size
    sim.response_owner_slots=actual


def complete_feedback(sim,observations,witnesses):
    """Observe only finished pilot tasks, then publish their suffix responses.

    Each checkpoint retains winner IDs and running maximum repairs, rather
    than all score vectors. Every later FP32 semantic score observes those
    winners. No response is valid until the terminal observation and the
    control-bank commit have completed. All work uses Program's shared units.
    """
    if getattr(sim,'feedback_published',False):raise ValueError('Pilot responses already published')
    if not hasattr(sim,'pilot_main_completed_cycle'):raise ValueError('Pilot main tasks are unfinished')
    begin=sim.g.maps.get('observation_begin',0);checkpoints=sim.g.maps['checkpoints']
    selected=np.flatnonzero(witnesses);count=len(selected);classes=observations['scores'].shape[-1]
    horizon=sim.c.get('correction_horizon','suffix')
    if horizon not in ('suffix','final'):raise ValueError(horizon)
    observed=list(range(begin,len(checkpoints)))
    if sim.c.get('diagnostic_feedback_actions_only',False):
        action=sim.c.get('predictor_first_exit')
        if (not sim.c.get('diagnostic_feedback_gate',False) or horizon!='final' or
                action!=sim.c.get('predictor_last_exit') or action is None or
                not begin<=action<len(checkpoints)-1):
            raise ValueError('Action-only pilot feedback requires a gated single final-horizon action')
        observed=[action,len(checkpoints)-1]
    start=sim.cycle;prior=None;numeric_winners=[];numeric_response=[]
    for depth in observed:
        if not count:break
        cp=checkpoints[depth];p=Program(sim,depth)
        source,weight,inner=readout(p,cp,selected)
        semantic=language(p,source,weight,inner)
        text=p.text(cp['semantic_width'],classes)
        score=p.linear(semantic,text)
        peak=p.tensor((count,2))
        p.event([score],peak,'aten.mul.Tensor',scalar_ops_per_row=2*(classes-1))
        width=math.ceil(2*(len(numeric_winners)+1)*4/SECTOR_BYTES)*SECTOR_BYTES//4
        state=p.tensor((count,width))
        inputs=[score,peak]+([] if prior is None else [p.refer(prior)])
        # Per historical winner: index/AGU, select score, subtract peak,
        # compare and max. Insert current winner and zero initial correction.
        update=len(numeric_winners) if horizon=='suffix' or depth==len(checkpoints)-1 else 0
        p.event(inputs,state,'aten.mul.Tensor',scalar_ops_per_row=5*update+2)
        before=sim.memory.counts.copy();flops=sim.stats['matrix_flops'];vectors=sim.stats['vector_cycles']
        mappings(sim,count*max(1,len(cp['readout_maps'])))
        p.run([state])
        if sim.c.get('diagnostic_row_windows',False):release_language_pack(sim,depth)
        if prior is not None:retire(sim,prior)
        prior=state
        sim.stats['diagnostic_flops']+=sim.stats['matrix_flops']-flops
        sim.stats['diagnostic_vector_ops']+=(sim.stats['vector_cycles']-vectors)*sim.c['vector_lanes']
        sim.stats['diagnostic_bytes']+=int((sim.memory.counts[:2]-before[:2]).sum())
        sim.stats['feedback_observed_rows']+=count
        sim.stats['feedback_terminal_observed_rows']+=count*int(depth==len(checkpoints)-1)
        if depth<len(checkpoints)-1:sim.stats['diagnostic_leaf_observations']+=count
        # Numerical companion uses the captured FP32 scores of completed
        # pilots only. It checks the response algorithm, not matrix RTL.
        scores=observations['scores'][depth,selected].astype(np.float32)
        for i,winner in enumerate(numeric_winners):
            if horizon=='suffix' or depth==len(checkpoints)-1:
                correction=scores.max(1)-scores[np.arange(count),winner]
                numeric_response[i]=np.maximum(numeric_response[i],correction)
        numeric_winners.append(observations['winner'][depth,selected])
        numeric_response.append(np.zeros(count,np.float32))
    # Banks store one value/valid pair per (context, checkpoint). Initialization
    # and capacity reservation happened before any B metadata was admitted.
    ports=sim.c['cam_lookups_per_cycle'];context=sim.g.maps['response_context']
    contexts=np.asarray(context['mapping'])[selected]
    if prior is not None:
        ready=sim.memory.access(np.arange(prior['address']//SECTOR_BYTES,
            math.ceil((prior['address']+prior['bytes'])/SECTOR_BYTES)))
        sim.cycle=max(sim.cycle,ready)
    from .predictor import (response_record_words,response_records,
                            selector_owner_addresses,selector_owner_bytes)
    slots=sim.c.get('witnesses_per_context',1);fields=response_record_words(sim.c)
    selected_owners=uses_witness_selector(sim.c)
    if selected_owners and not hasattr(sim,'response_owner_slots'):
        raise ValueError('Pilot owner selection has not completed')
    slot_ids=np.zeros(count,np.int64)
    if selected_owners:
        owner=sim.response_owner_slots[contexts]
        matches=owner==selected[:,None]
        if not matches.any(1).all():raise ValueError('Completed producer has no owner slot')
        slot_ids=matches.argmax(1)
        owner_address=selector_owner_addresses(contexts[:,None],np.arange(slots),
            rows(context['spec']),len(context['mapping']),sim.c)
        sim.cycle+=control_rounds(owner_address,ports)
        sim.stats['control_sram_read_bytes']+=count*slots*selector_owner_bytes(len(context['mapping']))
    table_cycles=0
    for depth in observed[:-1]:
        keys=(contexts*len(checkpoints)+depth)*slots+slot_ids
        # Each bank services 32 bits/cycle. A value/valid row needs two
        # read beats and two write beats; vector comparison is charged below.
        if fields==2 and slots==1:
            table_cycles+=4*control_rounds(keys,ports)+sim.c['cam_latency_cycles']
        else:
            addresses=keys[:,None]*fields+np.arange(fields)
            table_cycles+=2*control_rounds(addresses,ports)+sim.c['cam_latency_cycles']
    record_ops=3 if sim.c.get('packed_response_records',False) else 1
    pack_cycles=math.ceil(count*max(0,len(observed)-1)*record_ops/sim.c['vector_lanes'])
    ready=sim.issue_vector(sim.cycle,pack_cycles)
    sim.stats['vector_cycles']+=pack_cycles
    if record_ops>1:sim.stats['response_record_pack_cycles']+=pack_cycles
    sim.cycle=max(ready,sim.cycle)+table_cycles;sim.memory.clock=max(sim.memory.clock,sim.cycle)
    entries=count*max(0,len(observed)-1)
    sim.stats['feedback_commit_cycles']+=table_cycles
    sim.stats['control_sram_read_bytes']+=entries*4*fields;sim.stats['control_sram_write_bytes']+=entries*4*fields
    sim.stats['feedback_updates']+=entries
    sim.completed_response_values=np.asarray(numeric_response,np.float32)
    sim.completed_response_depths=observed
    sim.stats['feedback_skipped_pilot_observations']+=count*(len(checkpoints)-begin-len(observed))
    if slots>1 or sim.c.get('response_match_winner',False) or sim.c.get('predictor_conditioned_response',False):
        records=response_records(np.asarray(context['mapping']),witnesses,
            np.asarray(numeric_winners,np.int32),sim.completed_response_values,sim.c)
        keys,values,group=records
        # Gated stages are invalid, never zero repairs. Physical slot/bank
        # addresses above use actual context IDs, not this numeric compression.
        full_keys=np.full((len(checkpoints),*keys.shape[1:]),-1,keys.dtype)
        full_values=np.full(full_keys.shape,-np.inf,values.dtype)
        full_keys[observed]=keys;full_values[observed]=values
        sim.completed_response_records=(full_keys,full_values,group)
    sim.feedback_published=True;sim.feedback_completed_cycle=sim.cycle
    if prior is not None:retire(sim,prior)
    elapsed=sim.cycle-start;sim.stats['diagnostic_cycles']+=elapsed
    sim.stats['completed_feedback_cycles']+=elapsed
    if sim.c.get('predictor_joint_response',False):initialize_joint_scores(sim,observations)
    if sim.c.get('diagnostic_feedback_gate',False):initialize_feedback_gate(sim,witnesses)
    return elapsed


def initialize_joint_scores(sim,observations):
    """Commit only causally necessary owner queries through ordinary storage."""
    from .predictor import joint_context_actions,query_scores
    c=sim.c;depth=c['predictor_first_exit'];ports=c['cam_lookups_per_cycle']
    if not getattr(sim,'feedback_published',False):raise ValueError('Joint scores read unfinished feedback')
    if not c.get('diagnostic_feedback_gate',False):raise ValueError('Joint hardware requires the necessary feedback gate')
    keys,values,_=sim.completed_response_records;rb=c['predictor_response_bins']
    maximum=sim.predictor_response_actions[depth].reshape(-1,rb)
    nearest=sim.predictor_nearest_actions[depth].reshape(-1,rb)
    tolerance=min(c['semantic_tolerance'],c['class_repair_tolerance'])
    _,need=joint_context_actions(keys[depth],values[depth],sim.predictor_response_edges[depth],
        sim.predictor_nearest_edges[depth],maximum,nearest,tolerance)
    context=sim.g.maps['response_context'];actual=np.unique(context['mapping'])
    from .predictor import response_record_words
    count=rows(context['spec']);slots=c['witnesses_per_context'];fields=response_record_words(c)
    mapping=np.full((count,slots),-1,np.int32)
    if slots>2:
        # A short context consists entirely of complete pilots. No remaining
        # query can consume scores there; counts reside in the paid selector.
        population=np.bincount(context['mapping'],minlength=count)
        need=need&(population[actual]>slots)
    chosen=actual[need];owner_ids=[];cursor=0
    if getattr(sim,'feedback_gate_packet_owned',False):raise ValueError('Joint setup cannot borrow an active gate walker')
    start=max(sim.cycle,sim.memory.clock,sim.feedback_completed_cycle)
    # Clear the entire real-address-domain map; scan each complete owner set
    # and enumerate all frozen state cells. Tables and records share control
    # banks; no target-dependent fixed context count is charged instead.
    sim.cycle=start+math.ceil(mapping.nbytes/(4*ports))
    sim.stats['control_sram_write_bytes']+=mapping.nbytes
    # Borrow the already reserved, not-yet-active four-word gate walker.
    # PC/limit/valid/cursor are overwritten by normal gate initialization
    # only after this setup and every owner-score store complete.
    sim.cycle+=math.ceil(16/(4*ports))
    sim.stats['control_sram_write_bytes']+=16
    state_count=maximum.shape[0]
    for compressed,context_id in enumerate(actual):
        addresses=((context_id*len(sim.g.maps['checkpoints'])+depth)*slots+
                   np.arange(slots))[:,None]*fields+np.arange(fields)
        service=control_rounds(addresses,ports)+c['cam_latency_cycles']
        # Read both complete action planes and response thresholds once per
        # owner set; store slot row IDs only after predicates are evaluated.
        payload=4*(maximum.size+nearest.size+len(sim.predictor_response_edges[depth])+
                   len(sim.predictor_nearest_edges[depth]))
        service+=math.ceil(payload/(4*ports))+2*c['cam_latency_cycles']
        operations=slots*(4+len(sim.predictor_response_edges[depth])+len(sim.predictor_nearest_edges[depth]))
        operations+=state_count*(slots*5+4)
        if slots>2:
            # All equal-key subsets and their distinct max/nearest actions,
            # including the count>slots guard, use the existing vector lanes.
            operations+=slots*slots*3+state_count*slots*(slots+4)
            service+=control_rounds([context_id*2],ports)+c['cam_latency_cycles']
            sim.stats['control_sram_read_bytes']+=4
        arithmetic=(math.ceil(operations/c['vector_lanes'])+math.ceil(math.log2(state_count))+2+
                    max(0,math.ceil(math.log2(slots))-1))
        sim.cycle=sim.issue_vector(sim.cycle+service,arithmetic)
        sim.stats['vector_cycles']+=arithmetic
        sim.stats['control_sram_read_bytes']+=addresses.size*4+payload
        if need[compressed]:
            # The response key/repair/valid words do not contain native owner
            # IDs. Read the original selector plus the compact cursor only
            # after the static predicate returns, before generating addresses.
            from .predictor import selector_owner_addresses,selector_owner_bytes
            owner_address=selector_owner_addresses(context_id,np.arange(slots),count,
                len(context['mapping']),c)
            sim.cycle+=control_rounds(owner_address,ports)+c['cam_latency_cycles']
            sim.cycle+=math.ceil(1/ports)+c['cam_latency_cycles']
            sim.stats['control_sram_read_bytes']+=slots*selector_owner_bytes(len(context['mapping']))+4
            ids=sim.response_owner_slots[context_id]
            if np.any(ids<0):raise ValueError('An ambiguous context must own every fixed completed producer')
            if cursor+slots>c['queue_entries']*c['array_rows']:
                raise ValueError('Joint owner ID packet exceeds reserved gate capacity')
            mapping[context_id]=cursor+np.arange(slots)
            owner_ids.extend(ids);cursor+=slots
            allocation=math.ceil((2*slots+1)/c['vector_lanes'])+2
            sim.cycle=sim.issue_vector(sim.cycle,allocation)+math.ceil(1/ports)+math.ceil(slots/ports)
            sim.stats['vector_cycles']+=allocation
            sim.stats['control_sram_write_bytes']+=4+slots*4
        sim.cycle+=control_rounds(context_id*slots+np.arange(slots),ports)
        sim.stats['control_sram_write_bytes']+=slots*4
    owner_ids=np.asarray(owner_ids,np.int64)
    sim.cycle+=math.ceil(1/ports)
    sim.stats['control_sram_write_bytes']+=4
    sim.stats['joint_owner_selector_read_bytes']+=len(owner_ids)*4
    sim.stats['joint_owner_cursor_updates']+=len(chosen)
    sim.joint_owner_rows=mapping
    sim.stats['joint_owner_contexts']=len(chosen);sim.stats['joint_query_owners']=len(owner_ids)
    if len(owner_ids):
        # Native IDs persist in the already reserved bounded gate packet,
        # rather than an uncharged invocation-sized host list. It cannot be
        # reused by the ordinary walker until query mapping has consumed it.
        sim.feedback_gate_packet_owned=True
        sim.stats['feedback_gate_packet_peak_bytes']=max(sim.stats['feedback_gate_packet_peak_bytes'],len(owner_ids)*4)
        consume_gate_packet(sim,len(owner_ids))
        cp=sim.g.maps['checkpoints'][depth];p=Program(sim,depth)
        query,join,composition,*_=prepare_query(p,cp,owner_ids,observations['scores'].shape[-1],True)
        score=p.unary(p.normalize(query),element=2,op='aten.to.dtype')
        if composition is not None and c.get('diagnostic_overlap_composition',False):p.overlap_composition(composition,join)
        before=sim.memory.counts.copy();flops=sim.stats['matrix_flops'];vectors=sim.stats['vector_cycles']
        mappings(sim,len(owner_ids)*len(cp['readout_maps']))
        p.run([score])
        release_gate_packet(sim)
        # Ordinary feature directory owns the payload, with real writeback
        # before any eviction/reload. Numeric data is companion validation.
        sim.joint_score_payload=score
        sim.joint_numeric_scores=query_scores(observations)[0][depth,owner_ids].astype(np.float16).astype(np.float32)
        sim.stats['joint_owner_score_bytes']=score['bytes']
        sim.stats['diagnostic_flops']+=sim.stats['matrix_flops']-flops
        sim.stats['diagnostic_vector_ops']+=(sim.stats['vector_cycles']-vectors)*c['vector_lanes']
        sim.stats['diagnostic_bytes']+=int((sim.memory.counts[:2]-before[:2]).sum())
    # Publish the row IDs only after all score producer stores complete. It
    # is invocation-local and remains valid through the final consumer join.
    sim.cycle=max(sim.cycle,sim.memory.clock,sim.stats['last_tile_completion'])
    sim.joint_scores_ready=sim.cycle;sim.memory.clock=max(sim.memory.clock,sim.cycle)
    sim.stats['joint_owner_setup_cycles']+=sim.cycle-start
    sim.stats['diagnostic_cycles']+=sim.cycle-start


def joint_decision(p,observations,stops,control,query,query_rows,lut_keys):
    """Two finite lookups, then paid distance only for differing actions."""
    sim=p.sim;c=sim.c;depth=p.depth;ports=c['cam_lookups_per_cycle'];rb=c['predictor_response_bins']
    tolerance=min(c['semantic_tolerance'],c['class_repair_tolerance'])
    if not hasattr(sim,'joint_scores_ready'):raise ValueError('Joint owner scores are unpublished')
    keys,values,group=sim.completed_response_records
    score=observations.get('control_scores',observations['scores'])[depth,control].astype(np.float32)
    score/=np.maximum(np.linalg.norm(score,axis=1,keepdims=True),np.finfo(np.float32).tiny)
    match=keys[depth,group[control]]==score.argmax(1)[:,None]
    local=np.where(match,values[depth,group[control]],-np.inf)
    maximum=local.max(1)
    max_keys=lut_keys*rb+np.searchsorted(sim.predictor_response_edges[depth],maximum)
    maximum_action=sim.predictor_response_actions[depth][max_keys]&np.isfinite(maximum)&(maximum<=tolerance)
    nearest_keys=lut_keys[:,None]*rb+np.searchsorted(sim.predictor_nearest_edges[depth],local)
    actions=sim.predictor_nearest_actions[depth][nearest_keys]&np.isfinite(local)&(local<=tolerance)
    ambiguous=(~maximum_action)&np.any(actions&match,axis=1)&np.any((~actions)&match,axis=1)
    positions=np.flatnonzero(ambiguous)
    context=np.asarray(sim.g.maps['response_context']['mapping'])[control]
    owner_rows=sim.joint_owner_rows[context[positions]]
    if np.any(owner_rows<0):raise ValueError('Static owner predicate missed a legal distance request')
    nearest_action=actions.any(1)
    if len(positions):
        distances=np.abs(score[positions,None,:]-sim.joint_numeric_scores[owner_rows]).sum(2)
        distances=np.where(match[positions],distances,np.inf)
        nearest_action[positions]=actions[positions,distances.argmin(1)]
    np.testing.assert_array_equal(maximum_action|nearest_action,stops[control]==depth)
    slots=c['witnesses_per_context']
    flags=p.tensor((len(control),1),4,'index')
    flag_at=p.event([],flags,'aten.full.default',scalar_ops_per_row=slots*(4+len(sim.predictor_nearest_edges[depth]))+3*slots)
    producers=[p.g.events[flag_at]]
    if len(positions):
        payload=p.refer(sim.joint_score_payload)
        decision=p.tensor((len(positions),1),4,'index')
        at=p.event([query]+[payload]*slots+[flags],decision,'aten.mul.Tensor',
            diagnostic_control_rows=[np.asarray(query_rows)[positions]]+[owner_rows[:,slot] for slot in range(slots)]+[positions],
            # FP32 query normalization and every FP16 owner widen/distance,
            # followed by key masking, first-rank tie selection and branch.
            scalar_ops_per_row=(3+4*slots)*score.shape[-1]+6+slots+2*max(0,slots-2),
            # Two serial sum trees (norm then L1); square, sqrt, clamp,
            # divide, subtract, abs, compare, rank tie and action select.
            # All owner distances share lanes and run in parallel only
            # where the ordinary finite lane throughput permits it.
            scalar_dependency_cycles=(2*math.ceil(math.log2(max(1,score.shape[-1])))+9+
                                      max(0,math.ceil(math.log2(slots))-1)))
        producers.append(p.g.events[at])
    rounds=control_rounds(nearest_keys,ports)+c['cam_latency_cycles']
    map_rounds=(control_rounds(context[positions,None]*slots+np.arange(slots),ports)+c['cam_latency_cycles']
                if len(positions) else 0)
    sim.stats['joint_distance_rows']+=len(positions)
    sim.stats['joint_nearest_lookup_rows']+=len(control)*slots
    return dict(rounds=rounds,read_bytes=nearest_keys.size*4,
                map_rounds=map_rounds,map_read_bytes=owner_rows.size*4,producers=producers)


def execute(sim,observations,stops,witnesses,depths,phase):
    if sim.c.get('trajectory_space')!='query_scores':raise NotImplementedError('Finite diagnostic lowering currently supports query trajectories')
    slots=sim.c.get('witnesses_per_context',1)
    if ((sim.c.get('response_match_winner',False) or sim.c.get('predictor_conditioned_response',False))
        and (not 1<=slots<=8 or sim.c['witness_stride']!=1)):
        raise NotImplementedError('Finite conditioned feedback requires one to eight fixed owners in every context')
    model=getattr(sim.g,'trace',{}).get('model','EmbodiedSplat')
    if model!='EmbodiedSplat':
        if model not in ('SceneSplat','Chorus'):
            raise NotImplementedError('Unknown diagnostic model')
        from .readout import validate_recipe
        for cp in sim.g.maps['checkpoints']:
            if not cp.get('readout_identity',False):
                if 'readout_recipe' not in cp:
                    raise NotImplementedError('Recapture the affine readout recipe; legacy finite support was EmbodiedSplat only')
                validate_recipe(cp,model)
    # Pilots are not candidates for an early decision. Once their main tasks
    # have completed, complete_feedback observes their FP32 semantic scores,
    # including the terminal checkpoint, and only then publishes responses.
    if phase=='pilot':return 0
    if witnesses.any() and not getattr(sim,'feedback_published',False):
        raise ValueError('A decision cannot read unfinished pilot feedback')
    start=sim.cycle;classes=observations['scores'].shape[-1];emitted=[]
    for depth,cp in enumerate(sim.g.maps['checkpoints'][:-1]):
        if depth<sim.g.maps.get('observation_begin',0) or (depths is not None and depth not in depths):continue
        if depth>=sim.g.maps.get('observation_end',len(sim.g.maps['checkpoints'])-1):continue
        active=stops>=depth
        if phase=='pilot':active&=witnesses
        elif phase=='remaining':active&=~witnesses
        selected=np.flatnonzero(active)
        if not len(selected):continue
        windowed=sim.c.get('diagnostic_row_windows',False)
        window_rows=row_window_size(sim,classes) if windowed else len(stops)
        sim.stats['diagnostic_window_rows']=max(sim.stats['diagnostic_window_rows'],window_rows)
        active_selected=selected
        # Partition by the immutable output ID, rather than the changing rank
        # among active rows. A surviving leaf always finds its preceding state
        # in the same window, even after other leaves have ended.
        window_ids=np.unique(active_selected//window_rows) if windowed else [0]
        for window_id in window_ids:
            selected=(active_selected[active_selected//window_rows==window_id]
                      if windowed else active_selected)
            if sim.c.get('diagnostic_feedback_gate',False):selected=gated_observations(sim,selected,classes)
            if not len(selected):continue
            if sim.c.get('diagnostic_feedback_gate',False):consume_gate_packet(sim,len(selected))
            history_key=(phase,int(window_id)) if windowed else phase
            if windowed:
                sim.stats['diagnostic_windows']+=1
                sim.stats['control_sram_write_bytes']+=8
                sim.stats['control_sram_read_bytes']+=8
                sim.cycle=max(sim.cycle,sim.memory.clock)+2*math.ceil(2/sim.c['cam_lookups_per_cycle'])
            operands=list(zip(cp['readout_inputs'],cp['readout_maps']))
            representatives,inverse,checks=equivalent_rows(operands,selected)
            history=[]
            for prior in sim.g.maps['checkpoints'][max(sim.g.maps.get('observation_begin',0),depth-sim.c['predictor_window']+1):depth+1]:
                history.extend(zip(prior['readout_inputs'],prior['readout_maps']))
            context=sim.g.maps.get('response_context')
            if context is not None:history.append((context['spec'],context['mapping']))
            control,control_inverse,work=equivalent_rows(history,selected);checks+=work
            exits=selected[stops[selected]==depth];emitted.extend(exits.tolist())
            output_reps,_,work=equivalent_rows(operands,exits);checks+=work
            sim.stats['diagnostic_leaf_observations']+=len(selected)
            sim.stats['diagnostic_projected_rows']+=len(representatives)
            sim.stats['diagnostic_control_rows']+=len(control)
            p=Program(sim,depth);lang=cp['semantic_width']
            query,join,composition,language_weight,latent,query_source,query_inner,partitioned,half=prepare_query(
                p,cp,representatives,classes,windowed)
            direct_state=sim.c.get('diagnostic_direct_state',False)
            if not direct_state:query=p.unary(query,ops=3*classes)
            positions=np.full(len(stops),-1,np.int64);positions[selected]=inverse
            control_query=query if direct_state else p.gather(query,positions[control])
            state_inputs=[control_query];state_rows=[positions[control]]
            # Previous normalized score, a ring of w-1 scalar differences, the
            # window sum and observation count. Re-sum the bounded ring (+w ops)
            # so expired histories leave no FP32 rounding residue after a merge.
            # Incremental TV needs no redundant
            # copies of three language vectors and two full class vectors.
            matched=sim.c.get('response_match_winner',False)
            conditioned=sim.c.get('predictor_conditioned_response',False)
            slots=sim.c.get('witnesses_per_context',1)
            statewidth=trajectory_state_width(classes,sim.c)
            terminal_state=(sim.c.get('diagnostic_terminal_state',False) and direct_state and
                depth==sim.g.maps.get('observation_end',len(sim.g.maps['checkpoints'])-1)-1)
            # TV, margin, winner and valid feed this final action. No later
            # observation consumes its normalized score/ring history. Keep
            # the ordinary sector-aligned scratch owner through action commit.
            updated_width=(math.ceil(4*4/SECTOR_BYTES)*SECTOR_BYTES//4 if terminal_state else statewidth)
            if not hasattr(sim,'diagnostic_histories'):sim.diagnostic_histories={}
            prior=sim.diagnostic_histories.get(history_key)
            if prior is None:
                if not direct_state:
                    previous=p.tensor((len(control),statewidth));p.event([],previous,'aten.full.default')
            else:
                if prior['depth']!=depth-1:raise ValueError('TV observations must be consecutive')
                # The retained map selects the canonical state of each previous
                # active leaf. A split group reads that same completed state;
                # no per-leaf copy of the whole score/history vector is needed.
                positions_in_prior=np.searchsorted(prior['selected'],control)
                if not np.array_equal(prior['selected'][positions_in_prior],control):raise ValueError('A new leaf has no previous TV state')
                if direct_state:
                    state_inputs.extend([p.refer(prior['route']),p.refer(prior['state'])])
                    state_rows.extend([positions_in_prior,prior['inverse'][positions_in_prior]])
                else:
                    route=p.gather(p.refer(prior['route']),positions_in_prior)
                    previous=p.gather(p.refer(prior['state']),prior['inverse'][positions_in_prior])
                    # The route must return before the state-load address issues.
                    p.g.events[-2]['inputs']=[route]
                    p.g.events[-2]['op']='aten.mul.Tensor'
            updated=p.tensor((len(control),updated_width))
            if terminal_state:
                sim.stats['diagnostic_terminal_state_rows']+=len(control)
                sim.stats['diagnostic_terminal_history_elided_bytes']+=len(control)*(statewidth-updated_width)*4
            ops=5*classes+sim.c['predictor_window']+2+2*(sim.c['predictor_bins']-1)+6
            if direct_state:
                # Normalize this row in the state-update vector program,
                # before comparing it with the retained normalized history.
                # The same arithmetic remains paid; the wide normalized
                # query no longer needs a separate feature-store round trip.
                ops+=3*classes
                sim.stats['diagnostic_fused_normalization_rows']+=len(control)
            if sim.c.get('diagnostic_history_fp16',False):
                # Widen the preceding stored half query and round the new
                # normalized query for storage; TV accumulation stays FP32.
                casts=classes*(int(not terminal_state)+int(prior is not None))
                ops+=casts
                sim.stats['diagnostic_history_cast_elements']+=len(control)*casts
                sim.stats['diagnostic_history_packed_rows']+=len(control)
                if terminal_state:sim.stats['diagnostic_terminal_cast_elided_elements']+=len(control)*classes
            if matched:
                # First-index tie argmax and valid are produced by this same
                # state event; its stores explicitly precede table service.
                ops+=2*(classes-1)+3*slots
                sim.stats['diagnostic_winner_rows']+=len(control)
            elif slots>1:ops+=slots-1
            motion=sim.c.get('predictor_margin_trend',False) and depth==sim.c['predictor_first_exit']
            if motion:
                if prior is None or not direct_state or not sim.c.get('diagnostic_history_fp16',False):
                    raise ValueError('Boundary motion requires the actual preceding packed state')
                # Stable value/ID top2 reduction, two extractions from the
                # already loaded previous score lanes, two gaps, difference,
                # sign test and key-plane address. No new score/state buffer.
                ops+=6*(classes-1)+8
                sim.stats['diagnostic_boundary_motion_rows']+=len(control)
            if conditioned:ops+=sim.c['predictor_response_bins']-1+4
            if sim.c.get('predictor_query_rms',False):
                # Two FP32 multiplies per canonical state row, in this existing
                # vector program. Constants occupy eight paid control-SRAM bytes.
                ops+=2
                sim.stats['diagnostic_rms_rows']+=len(control)
                sim.stats['diagnostic_rms_scalar_ops']+=2*len(control)
            if direct_state:
                # Mapped score/history reads feed the existing state update;
                # no temporary score gather, route copy or state gather.
                current_map=positions[control]
                if not np.array_equal(current_map,np.arange(len(control))):
                    current_index=p.tensor((len(control),1),4,'index')
                    p.event([],current_index,'aten.full.default',scalar_ops_per_row=2)
                    state_inputs.append(current_index);state_rows.append(np.arange(len(control)))
                p.event(state_inputs,updated,'aten.mul.Tensor',scalar_ops_per_row=ops+2*len(state_inputs),
                        diagnostic_control=True,diagnostic_control_rows=state_rows,
                        diagnostic_normalize_query_input=True,
                        diagnostic_terminal_state=terminal_state,
                        diagnostic_history_fp16=sim.c.get('diagnostic_history_fp16',False))
                sim.stats['diagnostic_direct_state_rows']+=len(control)
            else:p.event([control_query,previous],updated,'aten.mul.Tensor',scalar_ops_per_row=ops,diagnostic_control=True)
            state_event=p.g.events[-1]
            if motion:
                state_event['scalar_dependency_cycles']=2*math.ceil(math.log2(max(1,classes)))+7
                # Host companion numeric state only; all dataflow work and
                # the final key are charged in this state event above.
                if not hasattr(sim,'diagnostic_boundary_motion'):
                    from .predictor import boundary_motion
                    sim.diagnostic_boundary_motion=boundary_motion(observations,sim.c,depth)
            joint=None
            if sim.c.get('predictor_joint_response',False) and depth==sim.c['predictor_first_exit']:
                from .predictor import demand_features
                if not hasattr(sim,'diagnostic_features'):sim.diagnostic_features=demand_features(observations,sim.c)
                tv,margin=sim.diagnostic_features;te,me=sim.predictor_edges[depth]
                lut_keys=np.searchsorted(te,tv[depth,control])*sim.c['predictor_bins']+np.searchsorted(me,margin[depth,control])
                if motion:lut_keys+=sim.diagnostic_boundary_motion[control]*sim.c['predictor_bins']**2
                joint=joint_decision(p,observations,stops,control,query,positions[control],lut_keys)
            route=p.tensor((len(selected),1),4,'index')
            p.event([],route,'aten.full.default',scalar_ops_per_row=2)
            control_event=p.g.events[-1]
            roots=[updated,route]
            if len(output_reps):
                output=sim.g.events[-1]['outputs'][0]
                multicast=(sim.c.get('diagnostic_stream_multicast',False) and
                           (not windowed or sim.c.get('diagnostic_window_multicast',False)))
                if multicast:
                    compact=windowed and sim.c.get('compact_dispatch_gather',False)
                    # The existing compact CSR accepts (representative, global
                    # output ID) pairs. Initialize both words of every actual
                    # exit; ordinary gather still reads this full payload.
                    # The uncompressed path retains its full N-row index.
                    index=p.tensor((len(exits),2) if compact else (rows(output),1),4,'index')
                    p.event([],index,'aten.full.default',scalar_ops_per_row=4 if compact else 2)
                output_half=cp.get('output_fp16',False)
                partitioned_output=cp.get('output_partitioned',False)
                if partitioned_output:
                    output_language=p.partitioned_language(cp,output_reps,language_weight)
                elif (output_half and cp.get('readout_extra_flops_per_row') and
                      sim.c.get('diagnostic_reuse_query_source',False)):
                    source_rows=positions[output_reps]
                    output_language=language(p,query_source,language_weight,query_inner,
                                             output_half,source_rows=source_rows)
                else:
                    if partitioned:output_source,_,_=readout(p,cp,output_reps)
                    else:output_source=p.gather(latent,positions[output_reps])
                    output_language=language(p,output_source,language_weight,query_inner,output_half)
                if output_half and language_weight is not None and not partitioned_output:
                    sim.stats['diagnostic_output_fp16_flops']+=2*len(output_reps)*query_inner*lang
                exit_reps,exit_inverse,_=equivalent_rows(operands,exits)
                if not np.array_equal(exit_reps,output_reps):raise AssertionError('Exit identity changed')
                if multicast:
                    mapping=np.zeros(rows(output),np.int64);mapping[exits]=exit_inverse
                    at=p.event([output_language,index],output,'aten.index.Tensor',row_selector=True,dispatch_ids=exits,
                               dispatch_complete=bool(windowed and sim.c.get('compact_dispatch_gather',False)
                                                      and sim.c.get('diagnostic_exact_dispatch',False)))
                    p.g.maps['indices'][at]=[mapping]
                    if windowed:sim.stats['diagnostic_window_multicast_rows']+=len(exits)
                else:
                    leaf_language=p.gather(output_language,exit_inverse)
                    index=p.tensor((len(exits),1),4,'index')
                    p.event([],index,'aten.full.default',scalar_ops_per_row=2)
                    at=p.event([output,leaf_language,index],output,'',kind='indexed_store',dispatch_rows=len(exits))
                    p.g.maps['indices'][at]=[exits]
                roots.append(output)
            if composition is not None and sim.c.get('diagnostic_overlap_composition',False):
                p.overlap_composition(composition,join)
            # Current winner/context address the existing response summary and
            # demand table; finite control banks serialize conflicting requests.
            contexts=np.asarray(context['mapping'])[control] if context is not None else control
            ports=sim.c['cam_lookups_per_cycle']
            response_keys=contexts*len(sim.g.maps['checkpoints'])+depth
            if hasattr(sim,'predictor_edges'):
                from .predictor import demand_features
                if not hasattr(sim,'diagnostic_features'):
                    sim.diagnostic_features=demand_features(observations,sim.c)
                tv,margin=sim.diagnostic_features;te,me=sim.predictor_edges[depth]
                lut_keys=np.searchsorted(te,tv[depth,control])*sim.c['predictor_bins']+np.searchsorted(me,margin[depth,control])
                if motion:lut_keys+=sim.diagnostic_boundary_motion[control]*sim.c['predictor_bins']**2
            else:
                # Synthetic microcode tests have no calibrated LUT. Conservatively
                # direct every request to one bank; production runs pass edges.
                lut_keys=np.zeros(len(control),np.int64)
            from .predictor import response_record_words
            fields=response_record_words(sim.c)
            if fields==2 and slots==1:
                rounds=2*control_rounds(response_keys,ports)
            else:
                addresses=((response_keys[:,None]*slots+np.arange(slots))[:,:,None]*fields+
                           np.arange(fields))
                rounds=control_rounds(addresses,ports)
            rounds+=control_rounds(lut_keys,ports)
            if conditioned:
                if not hasattr(sim,'predictor_response_edges'):
                    raise ValueError('Conditional demand thresholds were not installed')
                response=np.full(len(control),-np.inf,np.float32)
                if hasattr(sim,'completed_response_records'):
                    keys,values,group=sim.completed_response_records
                    local=values[depth,group[control]].copy()
                    if matched:
                        score=observations.get('control_scores',observations['scores'])[depth,control].astype(np.float32)
                        score/=np.maximum(np.linalg.norm(score,axis=1,keepdims=True),np.finfo(np.float32).tiny)
                        current=score.argmax(1)
                        local=np.where(keys[depth,group[control]]==current[:,None],local,-np.inf)
                    response=local.max(1)
                conditional_keys=lut_keys*sim.c['predictor_response_bins']+np.searchsorted(
                    sim.predictor_response_edges[depth],response)
                rounds+=control_rounds(conditional_keys,ports)
                sim.stats['diagnostic_conditional_lookup_rows']+=len(control)
            exclusion_bytes=0
            if uses_witness_selector(sim.c):
                if not hasattr(sim,'response_owner_slots'):raise ValueError('Pilot owners are unavailable')
                rounds+=control_rounds(selected//32,ports)
                exclusion_bytes=len(selected)*4
            table_cycles=rounds+math.ceil(len(selected)/ports)+2*sim.c['cam_latency_cycles']
            control_at=next(i for i,e in enumerate(p.g.events) if e is control_event)
            state_at=next(i for i,e in enumerate(p.g.events) if e is state_event)
            table_cycles+=math.ceil(DIAGNOSTIC_DECISION_DESCRIPTOR_BYTES/(4*ports))
            read_bytes=len(control)*4*(slots*fields+1+int(conditioned))+exclusion_bytes+DIAGNOSTIC_DECISION_DESCRIPTOR_BYTES
            barriers=p.g.maps.setdefault('diagnostic_control_barriers',{})
            if joint is None:
                barriers[control_at]=dict(cycles=table_cycles,
                    read_bytes=read_bytes,producers=[state_at,control_at])
            else:
                # LUT return precedes flags. Only flags reveal which owner
                # map addresses to read; their return precedes distance issue.
                # Final action waits for every real distance producer.
                producer_ids=[next(i for i,e in enumerate(p.g.events) if e is event) for event in joint['producers']]
                pre=rounds+joint['rounds']+sim.c['cam_latency_cycles']
                post=table_cycles-rounds-sim.c['cam_latency_cycles']
                barriers.update({
                    state_at:dict(cycles=pre,read_bytes=read_bytes+joint['read_bytes'],producers=[state_at]),
                    control_at:dict(cycles=post,read_bytes=0,producers=[state_at,control_at]+producer_ids)})
                if joint['map_rounds']:
                    p.g.maps['diagnostic_control_barriers'][producer_ids[0]]=dict(
                        cycles=joint['map_rounds'],read_bytes=joint['map_read_bytes'],producers=[producer_ids[0]])
            if windowed and sim.c.get('diagnostic_window_overlap',False) and len(output_reps):
                # The suffix is fully issued before this program returns.
                # Its data and outstanding work retain their ordinary SRAM,
                # operand, output and bank owners; only global unit draining
                # is avoided while the next independent window prepares.
                p.g.maps['diagnostic_async_begin']=control_at+1
            before=sim.memory.counts.copy();before_flops=sim.stats['matrix_flops'];before_vector=sim.stats['vector_cycles']
            # Read/validate causal mappings before issuing consumers of derived
            # indices. The finite DMA queue overlaps bursts; control write ports
            # serialize returned words with backpressure.
            mappings(sim,checks)
            if sim.c.get('predictor_query_rms',False):
                sim.cycle=max(sim.cycle,sim.memory.clock)+math.ceil(2/sim.c['cam_lookups_per_cycle'])
                sim.stats['control_sram_read_bytes']+=8
            p.run(roots)
            if sim.c.get('diagnostic_feedback_gate',False):release_gate_packet(sim)
            sim.diagnostic_histories[history_key]=dict(depth=depth,state=updated,route=route,
                selected=selected,inverse=control_inverse)
            sim.stats['diagnostic_canonical_state_rows']+=len(control)
            sim.stats['diagnostic_lineage_bytes']+=len(selected)*4
            if prior is not None:
                retire(sim,prior['state']);retire(sim,prior['route'])
            if windowed and (len(exits)==len(selected) or
                    depth==sim.g.maps.get('observation_end',len(sim.g.maps['checkpoints'])-1)-1):
                # These are current committed actions. No future target label
                # or unobserved stage is consulted to release the history.
                retire(sim,updated);retire(sim,route)
                del sim.diagnostic_histories[history_key]
            sim.stats['diagnostic_flops']+=sim.stats['matrix_flops']-before_flops
            sim.stats['diagnostic_vector_ops']+=(sim.stats['vector_cycles']-before_vector)*sim.c['vector_lanes']
            sim.stats['diagnostic_bytes']+=int((sim.memory.counts[:2]-before[:2]).sum())
            if half and not partitioned:
                factor_shapes=cp.get('query_factor_shapes')
                projection_inner=factor_shapes[0][1] if factor_shapes else query_inner
                query_flops=2*len(representatives)*projection_inner*classes
                if factor_shapes:
                    query_flops+=2*len(representatives)*factor_shapes[0][0]*factor_shapes[0][1]
                sim.stats['diagnostic_query_fp16_flops']+=query_flops
        if windowed:release_language_pack(sim,depth)
    if emitted:
        output=sim.g.events[-1]['outputs'][0];width=output['shape'][-1]*output['element_bytes']
        sim.stats['early_output_rows']+=len(emitted);sim.stats['early_output_bytes']+=len(emitted)*width
    elapsed=sim.cycle-start;sim.stats['diagnostic_cycles']+=elapsed
    sim.memory.clock=max(sim.memory.clock,sim.cycle)
    return elapsed


def release(sim):
    """No checkpoint consumer remains when the parent finishes its invocation."""
    for history in getattr(sim,'diagnostic_histories',{}).values():
        retire(sim,history['state']);retire(sim,history['route'])
    sim.diagnostic_histories={}
    for spec in getattr(sim,'diagnostic_assets',{}).values():retire(sim,spec)
