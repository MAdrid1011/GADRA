"""Integer scans, indexed updates and compaction on existing vector/SRAM units.

Captured integers describe address topology only. They cannot supply feature
values, future stop decisions, or a cache-hit assumption. Scratch belongs to
one existing output owner and is retained until the microprogram completes.
"""
import math
import numpy as np


SORT='aten.sort.default'


def capture(event,args,kwargs):
    op=event.get('op','');source=args[0] if args else None
    if source is None:return
    if op=='aten.scatter_.value' and len(args)>=4:
        dim,index,value=args[1:4]
        if (source.ndim==2 and source.is_contiguous() and dim in (1,-1)
            and index.shape==(source.shape[0],1)
            and str(index.dtype)=='torch.int64'
            and isinstance(value,(bool,int,float)) and len(event['inputs'])==2):
            # One scalar assignment per row; conservatively retain the
            # whole-row read/modify/write on existing vector/SRAM paths.
            event['row_local_scalar_scatter']=True
        return
    primitive=None;keys=None
    if (op=='aten._unique2.default' and source.ndim==1 and source.is_contiguous()
            and not source.is_floating_point() and not source.is_complex()):
        sorted_output=bool(args[1] if len(args)>1 else kwargs.get('sorted',True))
        if not sorted_output:return
        primitive='unique';keys=source
        event['unique_return_inverse']=bool(args[2] if len(args)>2 else kwargs.get('return_inverse',False))
        event['unique_return_counts']=bool(args[3] if len(args)>3 else kwargs.get('return_counts',False))
    elif (op==SORT or op.startswith('aten.argsort.')) and source.ndim==1:
        primitive='sort'
        if not source.is_floating_point():keys=source
        event['control_sort_descending']=bool(args[2] if len(args)>2 else kwargs.get('descending',False))
    elif op=='aten.cumsum.default' and source.ndim==1 and not source.is_floating_point():
        primitive='scan'
    elif op=='aten.nonzero.default' and source.ndim==1:
        primitive='compact';keys=source
    elif op=='aten.bucketize.Tensor' and source.ndim==args[1].ndim==1:primitive='bucketize'
    elif op=='aten.bincount.default' and source.ndim==1 and len(event['inputs'])==1:
        primitive='histogram';keys=source
    elif op=='aten.scatter_reduce_.two' and source.ndim==1:
        dim=args[1] if len(args)>1 else kwargs['dim']
        reduce=kwargs.get('reduce',args[4] if len(args)>4 else None)
        if dim in (0,-1) and reduce in ('amin','amax') and kwargs.get('include_self',True):
            primitive='scatter_reduce';keys=args[2]
    elif op=='aten.scatter_.src' and source.ndim==2 and source.is_contiguous():
        dim=args[1] if len(args)>1 else kwargs['dim']
        if dim in (1,-1) and args[2].shape==source.shape:
            primitive='scatter';keys=args[2]
    if primitive is None:return
    event['control_primitive']=primitive
    if primitive=='compact':
        # nonzero of a 1D predicate physically stores one linear ID per row.
        # A 1D view ensures one complete scan, rather than repeating the
        # global scan once per 64 compacted outputs.
        out=event['outputs'][0]
        out.update(shape=[math.prod(out['shape'])],stride=[1])
    if keys is not None:
        import torch
        with torch._C._DisableTorchDispatch():
            actual=keys.detach().cpu().numpy().astype(np.int64,copy=True)
            if primitive=='scatter':
                actual+=np.arange(source.shape[0])[:,None]*source.shape[1]
                event['control_full_overwrite']=np.array_equal(np.sort(actual,axis=None),np.arange(source.numel()))
            event['_index_values']=[actual.reshape(-1)]


def execute(sim,index,ready):
    event=sim.g.events[index];primitive=event['control_primitive']
    if primitive=='sort':
        from .feeder import consumer_sort
        return consumer_sort(sim,event,ready)
    c=sim.c;lanes=c['vector_lanes'];source=event['inputs'][0]
    count=math.prod(source['shape']);base=sim.accumulator_base(4)
    sim.stats['control_microprograms']+=1

    def vector(operations,drain=0):
        nonlocal ready
        cost=max(math.ceil(operations/lanes),drain)
        ready=sim.issue_vector(ready,cost)
        sim.stats['vector_cycles']+=cost
        sim.stats['integer_control_vector_cycles']+=cost

    def transfer(offset,size,write=False):
        nonlocal ready
        ready=sim.output_transfer(size,ready,write=write,offset=base+offset)

    def reserve(size):
        if size>sim.output_budget():raise ValueError('Integer control scratch exceeds one finite output owner')
        sim.stats['integer_control_peak_scratch_bytes']=max(sim.stats['integer_control_peak_scratch_bytes'],size)

    if primitive=='bucketize':
        boundaries=math.prod(event['inputs'][1]['shape'])
        levels=math.ceil(math.log2(boundaries+1))
        # A fixed binary-search compare/select tree; tables are ordinary
        # already loaded inputs and every search level uses existing lanes.
        vector(count*3*levels,3*levels)
        return ready
    if primitive in ('scan','compact'):
        # Two integer scan banks ping-pong. Even a compact empty output
        # consumes the complete predicate and executes its prefix scan.
        element=8 if primitive=='compact' else event['outputs'][0]['element_bytes']
        pitch=count*element;reserve(2*pitch)
        vector(count);transfer(0,pitch,True)
        origin=0;destination=pitch;shift=1
        while shift<count:
            transfer(origin,pitch);transfer(origin,(count-shift)*element)
            vector(count-shift);transfer(destination,pitch,True)
            origin,destination=destination,origin;shift*=2
        transfer(origin,pitch)
        if primitive=='compact':
            # Rank gives exact linear output addresses; one retained vector
            # of lane ranks is routed through the existing store arbiter.
            vector(3*count)
            emitted=event['outputs'][0]['bytes']
            # The dead scan bank owns compacted IDs until the ordinary
            # destination store consumes them; no full output in registers.
            transfer(destination,emitted,True);transfer(destination,emitted)
            sim.stats['integer_compaction_source_elements']+=count
        return ready
    keys=np.asarray(sim.g.maps['indices'][index][0],dtype=np.int64).reshape(-1)
    if primitive=='histogram':
        bins=math.prod(event['outputs'][0]['shape']);element=event['outputs'][0]['element_bytes']
    else:
        bins=count;element=source['element_bytes']
    if np.any(keys<0) or np.any(keys>=bins):raise ValueError('Captured indexed control address outside destination')
    payload=bins*element;reserve(payload+lanes*8)
    if primitive!='scatter' or not event.get('control_full_overwrite',False):
        vector(bins);transfer(0,payload,True)
    banks=c['sram_banks']
    from .protocol import SECTOR_BYTES
    for begin in range(0,len(keys),lanes):
        local=keys[begin:begin+lanes]
        # Keys arrive through the ordinary source reads. A bounded scratch
        # vector retains them across counter bank conflicts.
        transfer(payload,len(local)*8,True);transfer(payload,len(local)*8)
        vector(2*len(local))
        addresses=base+local*element
        bank=(addresses//SECTOR_BYTES)%banks
        rank=np.argsort(bank,kind='stable');ordered=bank[rank]
        starts=np.r_[0,np.flatnonzero(np.diff(ordered))+1]
        ends=np.r_[starts[1:],len(local)]
        rounds=int(max(ends-starts,default=0))
        for at in range(rounds):
            chosen=rank[starts[ends-starts>at]+at]
            admitted=local[chosen]
            # Same-bank/word updates serialize, including repeated keys.
            loaded=ready
            if primitive!='scatter':
                for target in admitted:
                    loaded=max(loaded,sim.output_transfer(element,ready,offset=base+int(target)*element))
            ready=loaded;vector(len(admitted)*(1 if primitive=='histogram' else 2))
            stored=ready
            for target in admitted:
                stored=max(stored,sim.output_transfer(element,ready,write=True,offset=base+int(target)*element))
            ready=stored
        sim.stats['integer_control_conflict_rounds']+=rounds
    transfer(0,payload)
    return ready
