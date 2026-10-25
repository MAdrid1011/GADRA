"""Capacity fallbacks on existing vector lanes, output owners and hierarchy.

Large integer programs stream finite packets. Radix sort uses two off-chip
key/index arrays and byte-wide counting passes; scan carries one integer
between packets; scatter writes its real destination through the hierarchy.
All arrays here contain integer address topology, never semantic features.
"""
import math
import numpy as np

from . import memory_core as core
from .feeder import window_transfer,operand_capacity
from .graph import key
from .protocol import SECTOR_BYTES


RECORD_BYTES=16  # Native int64 key and stable int64 original index.
RADIX_BITS=8     # One byte per counting pass; 256 ordinary SRAM counters.
COUNTERS=1<<RADIX_BITS


def required(event):
    primitive=event.get('control_primitive')
    count=math.prod(event['inputs'][0]['shape'])
    if primitive=='sort':return 2*(1<<(max(1,count)-1).bit_length())*RECORD_BYTES
    if primitive in ('scan','compact'):return 2*count*event['outputs'][0]['element_bytes']
    if primitive in ('scatter','scatter_reduce','histogram'):
        return math.prod(event['outputs'][0]['shape'])*event['outputs'][0]['element_bytes']
    return 0


def needed(event,config,budget=None):
    if budget is None:
        budget=(operand_capacity(config,4)[2] if config.get('finite_operand_feeder',False)
                else config['output_sram_bytes']//config['output_buffers'])
    if event.get('control_primitive')=='unique':return True
    size=required(event)
    if event.get('control_primitive') in ('scatter','scatter_reduce','histogram'):
        size+=config['vector_lanes']*8
    return size>budget


@core.njit(cache=True)
def counter_updates(digit,values,base,ready,state,banks,calendar,lengths,bank_width,counts):
    """One bounded key packet; same-bank counter updates serialize in order."""
    addresses=base+digit*8
    bank=(addresses//SECTOR_BYTES)%len(banks)
    order=np.argsort(bank,kind='mergesort')
    ordered=bank[order]
    changes=np.flatnonzero(ordered[1:]!=ordered[:-1])+1
    starts=np.empty(len(changes)+1,np.int64);starts[0]=0;starts[1:]=changes
    ends=np.empty_like(starts);ends[:-1]=starts[1:];ends[-1]=len(digit)
    rounds=0
    for at in range(len(starts)):
        rounds=max(rounds,ends[at]-starts[at])
    positions=np.empty(len(digit),np.int64)
    for at in range(rounds):
        loaded=ready
        for group in range(len(starts)):
            if ends[group]-starts[group]<=at:continue
            lane=order[starts[group]+at]
            loaded=max(loaded,window_transfer(addresses[lane],8,ready,state,banks,calendar,lengths,bank_width,counts,False))
            positions[lane]=values[digit[lane]];values[digit[lane]]+=1
        ready=loaded+1  # Counter increment plus remaining-lane mask update.
        stored=ready
        for group in range(len(starts)):
            if ends[group]-starts[group]<=at:continue
            lane=order[starts[group]+at]
            stored=max(stored,window_transfer(addresses[lane],8,ready,state,banks,calendar,lengths,bank_width,counts,True))
        ready=stored+1
    return positions,ready,2*rounds


class Stream:
    def __init__(self,sim,ready,*,counter_bytes=0):
        self.sim=sim;self.begin=max(ready,sim.vector);self.ready=self.begin;self.operations=0
        self.base=sim.accumulator_base(4);self.counter_bytes=counter_bytes
        self.width=min(sim.c['vector_lanes'],(sim.output_budget()-counter_bytes)//(2*RECORD_BYTES))
        if self.width<1:raise ValueError('No finite integer streaming packet fits')
        self.scratch=counter_bytes+2*self.width*RECORD_BYTES
        sim.stats['integer_control_peak_scratch_bytes']=max(sim.stats['integer_control_peak_scratch_bytes'],self.scratch)
        # Two ranges, loop cursor and opcode, all native controller words.
        descriptor=6*np.dtype(np.int64).itemsize
        if descriptor>sim.control_capacity():raise ValueError('Integer streaming descriptor exceeds control SRAM')
        sim.stats['integer_stream_descriptor_bytes']=max(sim.stats['integer_stream_descriptor_bytes'],descriptor)
        self.ready+=math.ceil(descriptor/(4*sim.c['cam_lookups_per_cycle']))+sim.c['cam_latency_cycles']

    def vector(self,operations,minimum=0):
        cost=max(math.ceil(operations/self.sim.c['vector_lanes']),minimum)
        self.ready=self.sim.issue_vector(self.ready,cost);self.operations+=cost

    def scratch_transfer(self,offset,size,write=False):
        self.ready=self.sim.output_transfer(size,self.ready,write=write,offset=self.base+offset)

    def packet(self,size,offset=None):
        if size>self.width*RECORD_BYTES:raise ValueError('Integer packet exceeds its finite owner')
        offset=self.counter_bytes if offset is None else offset
        if offset<self.counter_bytes or offset+size>self.scratch:
            raise ValueError('Integer packet overlaps counters or exceeds scratch')
        self.scratch_transfer(offset,size,True)
        self.scratch_transfer(offset,size)

    def address(self,spec):
        if not spec['contiguous']:raise NotImplementedError('External integer programs require contiguous views')
        return self.sim.g.bases[key(spec)]+spec['address']-self.sim.g.storage[spec['storage']]['address']

    def read(self,address,positions,width,packet_offset=None):
        positions=np.asarray(positions,dtype=np.int64)
        if len(positions)>self.width:raise ValueError('Too many integer packet records')
        sectors=np.unique(((address+positions[:,None]*width+np.arange(width))//SECTOR_BYTES).ravel())
        m=self.sim.memory;m.clock=max(m.clock,self.ready)
        self.ready=max(self.ready,m.access(sectors))
        self.packet(len(positions)*width,packet_offset)
        self.sim.stats['integer_stream_read_bytes']+=len(positions)*width

    def write(self,address,positions,width,packet_offset=None):
        positions=np.asarray(positions,dtype=np.int64)
        self.packet(len(positions)*width,packet_offset)
        byte=address+positions[:,None]*width+np.arange(width)
        sectors,inverse=np.unique(byte//SECTOR_BYTES,return_inverse=True)
        masks=np.zeros(len(sectors),np.uint64)
        np.bitwise_or.at(masks,inverse.ravel(),np.left_shift(np.uint64(1),(byte%SECTOR_BYTES).astype(np.uint64)).ravel())
        self.sim.memory.clock=max(self.sim.memory.clock,self.ready)
        self.ready=max(self.ready,self.sim.store_sectors(sectors,masks,self.ready))
        self.sim.stats['integer_stream_write_bytes']+=len(positions)*width

    def updates(self,digits,values):
        m=self.sim.memory;banks,calendar,lengths=m.ports(2)
        begun=max(self.ready,self.sim.vector)
        result,self.ready,work=counter_updates(digits,values,self.base,begun,m.state,
            banks,calendar,lengths,self.sim.c['bank_bytes_per_cycle'],m.counts)
        self.ready=int(self.ready)
        if self.sim.c.get('vector_ready_queue',False):
            end=self.sim.vector_queue.reserve(begun,self.ready-begun,begun)
            if end!=self.ready:raise AssertionError('Counter packet lost exclusive vector interval')
        self.sim.vector=max(self.sim.vector,self.ready)
        if self.sim.activity is not None:self.sim.activity.add_vector(begun,self.ready)
        self.operations+=int(work);return result

    def finish(self):
        sim=self.sim;self.ready=int(self.ready)
        sim.stats['vector_cycles']+=self.operations
        sim.stats['integer_control_vector_cycles']+=self.operations
        sim.stats['integer_stream_programs']+=1
        return self.ready


def radix(sim,index,ready,*,integer_values=None,return_records=False):
    event=sim.g.events[index];source=event['inputs'][0]
    values=np.asarray(sim.g.maps['indices'][index][0] if integer_values is None else integer_values).reshape(-1)
    if values.dtype.kind not in 'iu':raise ValueError('External radix requires captured integer keys')
    native_bytes=source['element_bytes'];count=len(values)
    if count!=math.prod(source['shape']):raise ValueError('Captured sort keys differ from input extent')
    # Two histograms permit the same ping-pong prefix scan as small scans.
    stream=Stream(sim,ready,counter_bytes=2*COUNTERS*8)
    buffers=[sim.allocate_workspace(count*RECORD_BYTES) for _ in range(2)]
    sim.stats['integer_stream_workspace_bytes']+=2*count*RECORD_BYTES
    order=np.arange(count,dtype=np.int64)
    # Captured integer topology supplies addresses only. Every counting,
    # prefix, cursor and scatter operation below is charged at runtime.
    transformed=values.astype(np.uint64)
    if source.get('dtype','torch.int64').startswith('torch.int'):
        transformed^=np.uint64(1)<<np.uint64(native_bytes*8-1)
    if event.get('control_sort_descending',False):transformed=~transformed
    for shift in range(0,native_bytes*8,RADIX_BITS):
        digit=((transformed[order]>>np.uint64(shift))&np.uint64(COUNTERS-1)).astype(np.int64)
        counters=np.zeros(COUNTERS,np.int64)
        stream.vector(COUNTERS);stream.scratch_transfer(0,COUNTERS*8,True)
        for begin in range(0,count,stream.width):
            ids=np.arange(begin,min(count,begin+stream.width))
            stream.read(stream.address(source) if not shift else buffers[0],ids,
                        native_bytes if not shift else RECORD_BYTES)
            stream.vector(3*len(ids));stream.updates(digit[ids],counters)
        # Inclusive tree scan, converted to exclusive cursors.
        current=0;other=COUNTERS*8;distance=1
        while distance<COUNTERS:
            stream.scratch_transfer(current,COUNTERS*8)
            stream.scratch_transfer(current,(COUNTERS-distance)*8)
            stream.vector(COUNTERS-distance);stream.scratch_transfer(other,COUNTERS*8,True)
            current,other=other,current;distance*=2
        stream.scratch_transfer(current,COUNTERS*8);stream.vector(COUNTERS)
        stream.scratch_transfer(0,COUNTERS*8,True)
        cursors=np.r_[0,np.cumsum(counters[:-1])]
        next_order=np.empty_like(order)
        for begin in range(0,count,stream.width):
            ids=np.arange(begin,min(count,begin+stream.width))
            stream.read(stream.address(source) if not shift else buffers[0],ids,
                        native_bytes if not shift else RECORD_BYTES)
            stream.vector((4 if not shift else 3)*len(ids))
            destination=stream.updates(digit[ids],cursors)
            stream.write(buffers[1],destination,RECORD_BYTES)
            next_order[destination]=order[ids]
        order=next_order;buffers.reverse()
        sim.stats['integer_stream_radix_passes']+=1
    sim.stats['consumer_sort_keys']+=count
    if return_records:return stream,order,buffers
    for begin in range(0,count,stream.width):
        ids=np.arange(begin,min(count,begin+stream.width));stream.read(buffers[0],ids,RECORD_BYTES)
        stream.vector(len(ids))
        for output in event['outputs']:stream.write(stream.address(output),ids,output['element_bytes'])
    # The dead temporary arrays have no consumers. Dirty eviction traffic
    # already paid remains; final temporary contents need no writeback.
    for address in buffers:
        sim.memory.discard(np.arange(address//SECTOR_BYTES,math.ceil((address+count*RECORD_BYTES)/SECTOR_BYTES)),stream.ready)
    return stream.finish(),order


def unique(sim,index,ready):
    """External unique/rank using real key/index records and a rank array."""
    event=sim.g.events[index];source=event['inputs'][0];output=event['outputs'][0]
    native=np.asarray(event['diagnostic_unique_native'],np.int64).reshape(-1)
    if (len(native)!=math.prod(source['shape']) or np.any(native<0) or
            np.any(native>=event['diagnostic_unique_rows']) or output['element_bytes']!=4):
        raise ValueError('External native unique topology differs from its input domain')
    stream=None;buffers=[];ordered=False
    if sim.c.get('diagnostic_sorted_unique',False):
        # Verify the entire current key stream, one bounded packet at a time.
        # A failed probe pays its reads/comparisons before falling back.
        stream=Stream(sim,ready,counter_bytes=16);ordered=True
        stream.vector(2);stream.scratch_transfer(0,16,True)
        previous=-1
        for begin in range(0,len(native),stream.width):
            ids=np.arange(begin,min(len(native),begin+stream.width))
            stream.read(stream.address(source),ids,source['element_bytes'])
            stream.scratch_transfer(0,16);stream.vector(3*len(ids)+2)
            stream.scratch_transfer(0,16,True)
            if (len(ids) and (native[ids[0]]<previous or
                    np.any(native[ids[1:]]<native[ids[:-1]]))):
                ordered=False;break
            if len(ids):previous=int(native[ids[-1]])
        sim.stats['diagnostic_unique_stream_probe_keys']+=begin+len(ids) if len(native) else 0
        if not ordered:ready=stream.finish();stream=None
    if ordered:
        order=np.arange(len(native));records=stream.address(source);record_bytes=source['element_bytes']
        sim.stats['diagnostic_unique_stream_ordered_programs']+=1
    else:
        stream,order,buffers=radix(sim,index,ready,integer_values=native,return_records=True)
        records=buffers[0];record_bytes=RECORD_BYTES
    values=native[order];different=np.r_[True,values[1:]!=values[:-1]] if len(values) else np.zeros(0,bool)
    ranks=np.cumsum(different)-1;used=values[different]
    if math.prod(output['shape'])!=len(used)+len(native):
        raise ValueError('External native unique output extent differs')
    rank_address=sim.allocate_workspace(len(native)*4)
    sim.stats['integer_stream_workspace_bytes']+=len(native)*4
    stream.vector(2);stream.scratch_transfer(0,16,True)
    # Keep the key/index packet plus two int64 scan banks in the same
    # declared 32-byte-per-lane scratch. Recycle the last bank for stores.
    store_packet=stream.counter_bytes+stream.width*24
    for begin in range(0,len(native),stream.width):
        ids=np.arange(begin,min(len(native),begin+stream.width))
        stream.read(records,ids,record_bytes)
        stream.scratch_transfer(0,16);stream.vector(3*len(ids))
        origin=stream.counter_bytes+stream.width*16;destination=origin+stream.width*8
        stream.scratch_transfer(origin,len(ids)*8,True);distance=1
        while distance<len(ids):
            stream.scratch_transfer(origin,len(ids)*8)
            stream.scratch_transfer(origin,(len(ids)-distance)*8)
            stream.vector(len(ids)-distance);stream.scratch_transfer(destination,len(ids)*8,True)
            origin,destination=destination,origin;distance*=2
        stream.scratch_transfer(origin,len(ids)*8);stream.vector(len(ids)+2)
        stream.scratch_transfer(0,16,True)
        stream.write(rank_address,ids,4,packet_offset=store_packet)
    # The completed prefix carry now supplies the real unique count, hence
    # the packed inverse offset. A metadata size alone never supplies it.
    stream.scratch_transfer(0,16);stream.vector(2)
    for begin in range(0,len(native),stream.width):
        ids=np.arange(begin,min(len(native),begin+stream.width))
        stream.read(records,ids,record_bytes)
        stream.read(rank_address,ids,4,packet_offset=store_packet)
        stream.scratch_transfer(0,16);stream.vector(3*len(ids)+2)
        stream.scratch_transfer(0,16,True)
        selected=ids[different[ids]]
        stream.write(stream.address(output),ranks[selected],4,packet_offset=store_packet)
        stream.read(rank_address,ids,4,packet_offset=store_packet)
        stream.vector(2*len(ids))
        stream.write(stream.address(output),len(used)+order[ids],4,packet_offset=store_packet)
    inverse=np.empty(len(native),np.int64);inverse[order]=ranks
    for address,size in [(a,len(native)*RECORD_BYTES) for a in buffers]+[(rank_address,len(native)*4)]:
        sim.memory.discard(np.arange(address//SECTOR_BYTES,math.ceil((address+size)/SECTOR_BYTES)),stream.ready)
    sim.stats['diagnostic_unique_stream_programs']+=1
    sim.stats['diagnostic_unique_programs']+=1
    return stream.finish(),used,inverse


def unique2(sim,index,ready):
    """Sorted integer unique, inverse and counts through bounded packets.

    Captured keys determine routing only. Two paid radix record arrays,
    an int64 rank array, packet prefix scans and actual output stores build
    every result. The dynamic run count is obtained by the completed scan.
    """
    event=sim.g.events[index];source=event['inputs'][0]
    native=np.asarray(sim.g.maps['indices'][index][0]).reshape(-1)
    outputs=event['outputs'];count=len(native)
    if count!=math.prod(source['shape']) or len(outputs)!=3:
        raise ValueError('Integer unique source/output contract differs')
    if not count:
        if any(s['bytes'] for s in outputs):raise ValueError('Empty unique output extent differs')
        stream=Stream(sim,ready,counter_bytes=32);stream.vector(1)
        sim.stats['integer_unique_programs']+=1
        return stream.finish(),native.copy(),np.empty(0,np.int64),np.empty(0,np.int64)
    stream,order,buffers=radix(sim,index,ready,return_records=True)
    values=native[order];different=np.r_[True,values[1:]!=values[:-1]]
    ranks=np.cumsum(different)-1;used=values[different]
    ends=np.r_[different[1:],True]
    counts=np.diff(np.r_[np.flatnonzero(different),count])
    expected=[len(used),count if event['unique_return_inverse'] else 0,
              len(used) if event['unique_return_counts'] else 0]
    if [math.prod(s['shape']) for s in outputs]!=expected:
        raise ValueError('Integer unique output extents differ')
    if outputs[0]['element_bytes']!=source['element_bytes'] or any(s['element_bytes']!=8 for s in outputs[1:]):
        raise ValueError('Integer unique output dtype contract differs')
    rank_address=sim.allocate_workspace(count*8)
    sim.stats['integer_stream_workspace_bytes']+=count*8
    origin_base=stream.counter_bytes+stream.width*16
    store_packet=stream.counter_bytes+stream.width*24
    # Carry words: previous key, rank, validity and current run start.
    stream.vector(4);stream.scratch_transfer(0,32,True)
    for begin in range(0,count,stream.width):
        ids=np.arange(begin,min(count,begin+stream.width))
        stream.read(buffers[0],ids,RECORD_BYTES)
        stream.scratch_transfer(0,32);stream.vector(3*len(ids))
        origin=origin_base;destination=origin+stream.width*8
        stream.scratch_transfer(origin,len(ids)*8,True);distance=1
        while distance<len(ids):
            stream.scratch_transfer(origin,len(ids)*8)
            stream.scratch_transfer(origin,(len(ids)-distance)*8)
            stream.vector(len(ids)-distance)
            stream.scratch_transfer(destination,len(ids)*8,True)
            origin,destination=destination,origin;distance*=2
        stream.scratch_transfer(origin,len(ids)*8);stream.vector(len(ids)+3)
        stream.scratch_transfer(0,32,True)
        stream.write(rank_address,ids,8,packet_offset=store_packet)
    stream.scratch_transfer(0,32);stream.vector(2)
    for begin in range(0,count,stream.width):
        ids=np.arange(begin,min(count,begin+stream.width))
        stream.read(buffers[0],ids,RECORD_BYTES)
        stream.read(rank_address,ids,8,packet_offset=store_packet)
        stream.vector(2*len(ids))
        starts=ids[different[ids]]
        stream.write(stream.address(outputs[0]),ranks[starts],source['element_bytes'],packet_offset=store_packet)
        if event['unique_return_inverse']:
            stream.read(rank_address,ids,8,packet_offset=store_packet)
            stream.vector(len(ids))
            stream.write(stream.address(outputs[1]),order[ids],8,packet_offset=store_packet)
        if event['unique_return_counts']:
            # Prefix-max of run starts, seeded by the preceding packet's
            # carry. Run ends subtract this start from their global position.
            # No host run length is supplied to the execution datapath.
            if ids[-1]+1<count:
                stream.read(buffers[0],np.array([ids[-1]+1]),RECORD_BYTES,packet_offset=origin_base)
                stream.vector(2)
            stream.scratch_transfer(0,32);stream.vector(3*len(ids))
            origin=origin_base;destination=origin+stream.width*8
            stream.scratch_transfer(origin,len(ids)*8,True);distance=1
            while distance<len(ids):
                stream.scratch_transfer(origin,len(ids)*8)
                stream.scratch_transfer(origin,(len(ids)-distance)*8)
                stream.vector(len(ids)-distance)
                stream.scratch_transfer(destination,len(ids)*8,True)
                origin,destination=destination,origin;distance*=2
            stream.scratch_transfer(origin,len(ids)*8);stream.vector(2*len(ids)+2)
            stream.scratch_transfer(0,32,True)
            # The count scan can overwrite packet scratch, so reload ranks
            # before routing completed run counts to their actual rows.
            stream.read(rank_address,ids,8,packet_offset=store_packet)
            selected=ids[ends[ids]]
            stream.vector(len(ids))
            stream.write(stream.address(outputs[2]),ranks[selected],8,packet_offset=store_packet)
    inverse=np.empty(count,np.int64);inverse[order]=ranks
    for address,size in [(a,count*RECORD_BYTES) for a in buffers]+[(rank_address,count*8)]:
        sim.memory.discard(np.arange(address//SECTOR_BYTES,math.ceil((address+size)/SECTOR_BYTES)),stream.ready)
    sim.stats['integer_unique_programs']+=1
    sim.stats['integer_unique_source_elements']+=count
    return stream.finish(),used,inverse,counts


def scatter(sim,index,ready):
    event=sim.g.events[index]
    if event['control_primitive']!='scatter' or not event.get('control_full_overwrite',False):
        raise NotImplementedError('External indexed fallback currently requires a complete overwrite permutation')
    stream=Stream(sim,ready);output=event['outputs'][0]
    output_address=stream.address(output)
    for source in event['inputs'][1:]:
        address=stream.address(source)
        if address<output_address+output['bytes'] and output_address<address+source['bytes']:
            raise NotImplementedError('External permutation inputs must not alias their destination')
    keys=np.asarray(sim.g.maps['indices'][index][0],np.int64).reshape(-1)
    if not np.array_equal(np.sort(keys),np.arange(math.prod(output['shape']))):
        raise ValueError('Complete overwrite metadata is not a permutation')
    for begin in range(0,len(keys),stream.width):
        ids=np.arange(begin,min(len(keys),begin+stream.width))
        # Destination's previous contents are dead. Index and value streams
        # remain ordinary actual input accesses, not captured data constants.
        for source in event['inputs'][1:]:stream.read(stream.address(source),ids,source['element_bytes'])
        stream.vector(3*len(ids));stream.write(output_address,keys[ids],output['element_bytes'])
    return stream.finish()


def scan(sim,index,ready):
    event=sim.g.events[index]
    if event['control_primitive']!='scan':raise NotImplementedError('External compaction requires its output-rank scatter')
    stream=Stream(sim,ready,counter_bytes=8)
    source=event['inputs'][0];output=event['outputs'][0];count=math.prod(source['shape'])
    # One retained scalar carry. Each packet's final sum advances it before
    # the owner is recycled; no invocation-sized prefix matrix is retained.
    stream.vector(1);stream.scratch_transfer(0,8,True)
    for begin in range(0,count,stream.width):
        ids=np.arange(begin,min(count,begin+stream.width))
        stream.read(stream.address(source),ids,source['element_bytes'])
        stream.vector(len(ids));stream.scratch_transfer(8,len(ids)*8,True)
        origin=8;destination=8+stream.width*RECORD_BYTES;distance=1
        while distance<len(ids):
            stream.scratch_transfer(origin,len(ids)*8)
            stream.scratch_transfer(origin,(len(ids)-distance)*8)
            stream.vector(len(ids)-distance);stream.scratch_transfer(destination,len(ids)*8,True)
            origin,destination=destination,origin;distance*=2
        stream.scratch_transfer(origin,len(ids)*8);stream.scratch_transfer(0,8)
        stream.vector(len(ids)+1);stream.scratch_transfer(0,8,True)
        stream.write(stream.address(output),ids,output['element_bytes'])
    return stream.finish()


def compact(sim,index,ready):
    """Packet predicate scan, scalar rank carry, then actual linear-ID stores.

    Captured predicate bits describe this real causal program's address
    topology. Every predicate is read at runtime and all rank arithmetic is
    charged; they are not early-exit labels or a substitute for feature work.
    """
    event=sim.g.events[index];source=event['inputs'][0];output=event['outputs'][0]
    masks=sim.g.maps['indices'].get(index)
    if masks is None or len(masks)!=1:
        raise ValueError('External compaction requires captured runtime predicate topology')
    predicate=np.asarray(masks[0]).reshape(-1)!=0
    count=math.prod(source['shape'])
    if len(predicate)!=count or int(predicate.sum())!=math.prod(output['shape']):
        raise ValueError('Runtime predicate and compacted output extents differ')
    stream=Stream(sim,ready,counter_bytes=8)
    source_address=stream.address(source);output_address=stream.address(output)
    if (source_address<output_address+output['bytes'] and
            output_address<source_address+source['bytes']):
        raise NotImplementedError('External compaction input must not alias its destination')
    stream.vector(1);stream.scratch_transfer(0,8,True)
    emitted=0
    for begin in range(0,count,stream.width):
        ids=np.arange(begin,min(count,begin+stream.width))
        stream.read(source_address,ids,source['element_bytes'])
        stream.vector(len(ids));stream.scratch_transfer(8,len(ids)*8,True)
        origin=8;destination=8+stream.width*RECORD_BYTES;distance=1
        while distance<len(ids):
            stream.scratch_transfer(origin,len(ids)*8)
            stream.scratch_transfer(origin,(len(ids)-distance)*8)
            stream.vector(len(ids)-distance);stream.scratch_transfer(destination,len(ids)*8,True)
            origin,destination=destination,origin;distance*=2
        stream.scratch_transfer(origin,len(ids)*8);stream.scratch_transfer(0,8)
        # Add the carry to packet ranks, select live source IDs, and update
        # carry before the next packet can recycle this scratch owner.
        stream.vector(3*len(ids)+1);stream.scratch_transfer(0,8,True)
        selected=int(predicate[ids].sum())
        stream.write(output_address,np.arange(emitted,emitted+selected),output['element_bytes'])
        emitted+=selected
    sim.stats['integer_compaction_source_elements']+=count
    return stream.finish()


def execute(sim,index,ready):
    primitive=sim.g.events[index]['control_primitive']
    sim.stats['control_microprograms']+=1
    if primitive=='sort':return radix(sim,index,ready)[0]
    if primitive=='unique':return unique2(sim,index,ready)[0]
    if primitive=='scan':return scan(sim,index,ready)
    if primitive=='compact':return compact(sim,index,ready)
    return scatter(sim,index,ready)
