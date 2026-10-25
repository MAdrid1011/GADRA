"""Current-input BatchNorm: finite global statistics, then row-local affine.

The supply DAG still demands the entire input. Only the executor separates
that one global consumer from the independent output-row consumers. Feature,
output, vector and DMA resources are the same ones used by the main engine.
"""
import math
import numpy as np
from .protocol import SECTOR_BYTES,SLOT_BYTES
from .graph import rows


def current_statistics(event):
    return event.get('op')=='aten.batch_norm.default' and event.get('batch_norm_training') is True


def references(graph,index,selected,event=None):
    event=graph.events[index] if event is None else event
    if not current_statistics(event):return graph.inputs_for(index,selected)
    row_inputs=set(event.get('normalization_row_input_positions',()))
    return [(event['inputs'][0],selected)]+[(s,selected if position in row_inputs else
        np.arange(rows(s),dtype=np.int64)) for position,s in enumerate(event['inputs'][1:],1)]


def sectors(address,nbytes):
    return np.arange(address//SECTOR_BYTES,math.ceil((address+nbytes)/SECTOR_BYTES),dtype=np.int64)


def control_beats(words,ports):
    return int(np.bincount(np.asarray(words)%ports,minlength=ports).max(initial=0))


def allocate(sim,event):
    if event.get('has_running_statistics') is not False:
        raise NotImplementedError('Training BatchNorm running-statistic updates are not lowered')
    source=event['inputs'][0]
    if len(source['shape'])!=2:raise NotImplementedError('BatchNorm statistics require a captured [N,C] feature matrix')
    count,width=source['shape']
    payload=width*4;pitch=math.ceil(payload/SECTOR_BYTES)*SECTOR_BYTES
    base=sim.allocate_workspace(3*pitch)
    scratch=sectors(base,payload);mean=sectors(base+pitch,payload);inverse=sectors(base+2*pitch,payload)
    # All partial sums use FP32. A narrow input must still fit the fixed
    # FP32 staging pitch; it does not grant extra intermediate capacity.
    row_bytes=width*4
    chunk=sim.output_budget()//row_bytes
    if not chunk:raise ValueError('A BatchNorm row exceeds the finite output owner')
    return dict(source=source,count=count,width=width,payload=payload,pitch=pitch,
        scratch=scratch,mean=mean,inverse=inverse,chunk=chunk)


def initialize(sim,state,floor):
    cost=math.ceil(state['width']/sim.c['vector_lanes'])
    ready=sim.issue_vector(floor,cost);sim.stats['vector_cycles']+=cost
    sim.memory.clock=max(sim.memory.clock,floor)
    sim.memory.access(state['scratch'],write=True,ready=ready)
    return ready


def reduce_chunk(sim,state,ids,pass_id,floor):
    """Reserve a finite tree; return its horizon without advancing dispatch."""
    m=sim.memory;c=sim.c;width=state['width'];row_bytes=width*4
    owner=int(np.argmin(sim.buffer_ready));saved=sim.active_output_buffer
    sim.active_output_buffer=owner
    m.clock=max(m.clock,floor,int(sim.buffer_ready[owner]))
    loaded=m.access(sim.g.sectors(state['source'],ids))
    if state['source']['element_bytes']!=4:
        cast=math.ceil(len(ids)*width/c['vector_lanes'])
        loaded=sim.issue_vector(loaded,cast);sim.stats['vector_cycles']+=cast
    staged=sim.output_transfer(len(ids)*row_bytes,loaded,write=True)
    ready=max(staged,m.access(state['scratch']))
    if pass_id:ready=max(ready,m.access(state['mean']))
    # Identical canonical partition/tree for streaming and serial passes.
    # Subtract/square fuses into the first pair reduction.
    remaining=len(ids);first=True
    while remaining>1:
        ready=sim.output_transfer(remaining*row_bytes,ready)
        cost=2*math.ceil(remaining*width/c['vector_lanes']) if pass_id and first else 0
        cost+=math.ceil((remaining//2)*width/c['vector_lanes'])
        if remaining%2:cost+=math.ceil(width/c['vector_lanes'])
        ready=sim.issue_vector(ready,cost);sim.stats['vector_cycles']+=cost
        remaining=(remaining+1)//2;first=False
        ready=sim.output_transfer(remaining*row_bytes,ready,write=True)
    ready=sim.output_transfer(row_bytes,ready)
    cost=math.ceil(width/c['vector_lanes'])
    if pass_id and first:cost+=2*math.ceil(width/c['vector_lanes'])
    ready=sim.issue_vector(ready,cost);sim.stats['vector_cycles']+=cost
    committed=m.access(state['scratch'],write=True,ready=ready)
    sim.buffer_ready[owner]=max(int(sim.buffer_ready[owner]),staged,ready,committed)
    sim.active_output_buffer=saved
    return int(committed)


def finalize(sim,state,pass_id,last,floor):
    m=sim.memory;c=sim.c
    m.clock=max(m.clock,floor)
    loaded=m.access(state['scratch'])
    cost=(1 if pass_id==0 else 3)*math.ceil(state['width']/c['vector_lanes'])
    ready=sim.issue_vector(max(loaded,last),cost);sim.stats['vector_cycles']+=cost
    target=state['inverse'] if pass_id else state['mean']
    committed=m.access(target,write=True,ready=ready)
    sim.cycle=max(last,ready,committed);m.clock=max(m.clock,sim.cycle)


def start_stream(sim,index,live):
    """Admit one exact adjacent producer, before its ready window is built.

    Generated bits promise final full-row stores; ready clocks protect actual
    completion. Canonical chunks and merge order never depend on B's ordering.
    Unsupported aliases, partial phases and column producers retain two passes.
    """
    if not (sim.a or sim.b) or not sim.c.get('stream_batch_norm_mean',False):return
    if getattr(sim,'normalization_mean_stream',None) is not None:return
    if index+1>=len(sim.g.events) or not len(live[index+1]):return
    if any(cp['after_event']==index for cp in sim.g.maps.get('checkpoints',[])):return
    event=sim.g.events[index];following=sim.g.events[index+1]
    if event['kind'] in ('indexed_store','progressive_update'):return
    if not current_statistics(following) or following.get('has_running_statistics') is not False:return
    if rows(following['outputs'][0])<sim.c.get('training_batch_norm_fusion_min_rows',0):return
    if index+1 in getattr(sim,'normalization_stats',{}):return
    if len(event['outputs'])!=1:return
    out=event['outputs'][0];source=following['inputs'][0]
    fields=('storage','version','address','shape','stride','element_bytes')
    if (any(out[f]!=source[f] for f in fields) or not out['contiguous']
        or len(source['shape'])!=2 or source['shape'][-1]<=0):return
    count,width=source['shape'];chunk=sim.output_budget()//(width*4)
    if not chunk or count<=chunk:return
    if not np.array_equal(np.sort(live[index]),np.arange(count)):return
    if index in getattr(live,'carry',{}):return
    from .feeder import use_sparse_columns
    if use_sparse_columns(sim.c,event,count):return
    if event['kind']=='linear' and sim.c.get('dense_output_columns',0):
        oversized=any(s['bytes']>sim.c['weight_sram_bytes'] for s in event['inputs'] if s['role']=='weight')
        if not sim.c.get('dense_columns_oversized_only',False) or oversized:return
    chunks=math.ceil(count/chunk)
    # N-bit generated mask, one 64-bit completion per canonical chunk and a
    # 32-byte descriptor (source/version, next chunk, scratch owner and valid).
    size=math.ceil(count/32)*4+chunks*8+32
    if size>=sim.control_capacity():return
    sim.normalization_stream_bytes=size
    state=allocate(sim,following);m=sim.memory
    m.clock+=math.ceil(size/(sim.c['cam_lookups_per_cycle']*4))
    sim.stats['normalization_stream_control_write_bytes']+=size
    before=m.counts.copy();vector_before=sim.stats['vector_cycles']
    last=initialize(sim,state,max(sim.cycle,m.clock))
    sim.normalization_mean_stream=dict(index=index+1,state=state,
        seen=np.zeros(count,bool),ready=np.zeros(chunks,np.int64),next=0,last=last,
        vector_cycles=sim.stats['vector_cycles']-vector_before,
        dram_bytes=int((m.counts[:2]-before[:2]).sum()))
    sim.stats['normalization_streams']+=1


def generated(sim,index,ids,ready):
    stream=getattr(sim,'normalization_mean_stream',None)
    if stream is None or stream['index']!=index+1:return
    if stream['seen'][ids].any():raise AssertionError('A source row was generated twice')
    state=stream['state'];c=sim.c;m=sim.memory
    ports=c['cam_lookups_per_cycle'];chunk=state['chunk']
    touched=np.unique(ids//chunk)
    # Each row performs an actual generated-word RMW. Same-bank requests
    # serialize; completion maxima use two 32-bit words per touched chunk.
    seen_words=math.ceil(state['count']/32)
    ready_words=np.r_[seen_words+2*touched,seen_words+2*touched+1]
    cost=2*control_beats(ids//32,ports)+c['cam_latency_cycles']
    cost+=2*control_beats(ready_words,ports)
    m.clock+=cost
    sim.stats['normalization_stream_control_read_bytes']+=len(ids)*4+len(touched)*8
    sim.stats['normalization_stream_control_write_bytes']+=len(ids)*4+len(touched)*8
    sim.stats['normalization_stream_control_cycles']+=cost
    stream['seen'][ids]=True
    stream['ready'][touched]=np.maximum(stream['ready'][touched],ready)
    before=m.counts.copy();vector_before=sim.stats['vector_cycles']
    while stream['next']<len(stream['ready']):
        at=stream['next'];begin=at*chunk;end=min(state['count'],begin+chunk)
        first=begin//32;words=math.ceil(end/32)-first
        scan=np.r_[np.arange(first,first+words),seen_words+2*at,seen_words+2*at+1]
        cost=control_beats(scan,ports)+c['cam_latency_cycles']
        m.clock+=cost
        sim.stats['normalization_stream_control_read_bytes']+=words*4+8
        sim.stats['normalization_stream_control_cycles']+=cost
        if not stream['seen'][begin:end].all():break
        # Do not advance dispatch to the reduction horizon. The other owner
        # can issue later matrix chunks while this vector tree is reserved.
        floor=max(sim.cycle,m.clock,int(stream['ready'][at]))
        last=reduce_chunk(sim,state,np.arange(begin,end,dtype=np.int64),0,floor)
        stream['last']=max(stream['last'],last);stream['next']+=1
        sim.stats['normalization_stream_chunks']+=1
    stream['vector_cycles']+=sim.stats['vector_cycles']-vector_before
    stream['dram_bytes']+=int((m.counts[:2]-before[:2]).sum())


def prepare(sim,index,event):
    if not hasattr(sim,'normalization_stats'):sim.normalization_stats={}
    if index in sim.normalization_stats:return sim.normalization_stats[index]
    stream=getattr(sim,'normalization_mean_stream',None)
    if stream is not None and stream['index']!=index:
        raise AssertionError('A streamed mean survived its adjacent BatchNorm')
    state=stream['state'] if stream is not None else allocate(sim,event)
    m=sim.memory;c=sim.c;count=state['count'];payload=state['payload'];pitch=state['pitch']
    start=sim.cycle;before=m.counts.copy();vector_before=sim.stats['vector_cycles']
    for pass_id in (0,1):
        floor=max(sim.cycle,m.clock);sim.cycle=floor
        if pass_id==0 and stream is not None:
            if stream['next']!=len(stream['ready']) or not stream['seen'].all():
                raise AssertionError('Streamed mean requires every source row')
            last=stream['last']
        else:
            last=initialize(sim,state,floor)
            for begin in range(0,count,state['chunk']):
                ids=np.arange(begin,min(count,begin+state['chunk']),dtype=np.int64)
                last=max(last,reduce_chunk(sim,state,ids,pass_id,floor))
        finalize(sim,state,pass_id,last,floor)
    tags=math.ceil(pitch/SLOT_BYTES)
    retirement=math.ceil(tags/c['cam_lookups_per_cycle'])+c['cam_latency_cycles']
    m.clock=max(m.clock,sim.cycle)+retirement
    m.discard(state['scratch'],m.clock);sim.cycle=max(sim.cycle,m.clock)
    result=dict(mean=state['mean'],inverse=state['inverse'])
    sim.normalization_stats[index]=result
    sim.stats['normalization_global_passes']+=2
    sim.stats['normalization_input_rows']+=2*count
    sim.stats['normalization_statistic_bytes']+=2*payload
    sim.stats['normalization_statistics_cycles']+=sim.cycle-start
    sim.stats['normalization_statistics_vector_cycles']+=sim.stats['vector_cycles']-vector_before
    sim.stats['normalization_statistics_dram_bytes']+=int((m.counts[:2]-before[:2]).sum())
    sim.stats['normalization_retirement_cycles']+=retirement
    if stream is not None:
        # Streaming work is counted where issued, not by subtracting all
        # producer work from a shared vector or DRAM counter.
        sim.stats['normalization_statistics_vector_cycles']+=stream['vector_cycles']
        sim.stats['normalization_statistics_dram_bytes']+=stream['dram_bytes']
        sim.stats['normalization_stream_control_peak_bytes']=max(
            sim.stats['normalization_stream_control_peak_bytes'],sim.normalization_stream_bytes)
        sim.normalization_mean_stream=None;sim.normalization_stream_bytes=0
    return result


def release(sim):
    for stats in getattr(sim,'normalization_stats',{}).values():
        data=np.r_[stats['mean'],stats['inverse']]
        tags=len(np.unique(data//(SLOT_BYTES//SECTOR_BYTES)))
        cost=math.ceil(tags/sim.c['cam_lookups_per_cycle'])+sim.c['cam_latency_cycles']
        sim.memory.clock=max(sim.memory.clock,sim.cycle)+cost
        sim.memory.discard(data,sim.memory.clock);sim.cycle=sim.memory.clock
        sim.stats['normalization_retirement_cycles']+=cost
    sim.normalization_stats={}
