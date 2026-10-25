"""Finite operand feeders shared by linear and real-offset sparse products.

Output SRAM is partitioned into accumulator windows and two operand slots per
array. Each slot owns its bytes from bank read until its last K beat is consumed.
The compiled loops preserve every transaction and every padded matrix tile.
"""
import math
import numpy as np

from . import memory_core as core
from .protocol import SECTOR_BYTES,SLOT_SECTORS,DMA_BURST_BYTES
from .copyroute import mapping as copy_mapping


# One fixed microprogram descriptor per physical array. Fields are 64-bit
# cycle/address/controller words, charged against the existing control SRAM.
(S_A,S_B,S_M,S_N,S_K,S_READY,S_ROW,S_END,S_COLUMN,S_COMMIT,
 S_ASLOT,S_BSLOT,S_CURSOR,S_ISSUED,S_VALID,S_UNIT)=range(16)
STREAM_DESCRIPTOR_WORDS=len(range(16))


def stream_descriptor_bytes(config):
    return (config['arrays']*config.get('prefetch_microprogram_tiles',1)*STREAM_DESCRIPTOR_WORDS*np.dtype(np.int64).itemsize
        if config.get('interleave_operand_slices',False) else 0)


def input_latch_bytes(config):
    """Two fixed sector-wide input registers per array lane, paid from SRAM."""
    if not config.get('stream_operand_slices',False):return 0
    return config['arrays']*2*(config['array_rows']+config['array_cols'])*SECTOR_BYTES


def consumer_sort(sim,event,ready):
    """Stable bitonic key/index microcode in one existing output owner.

    Each 16-byte record holds the native key and 64-bit original index.
    Two power-of-two arrays ping-pong after every compare-exchange layer;
    both self and XOR-partner reads finish before the next layer writes.
    Original input reads and final permutation stores are paid by execute().
    """
    count=math.prod(event['inputs'][0]['shape'])
    padded=1<<(max(1,count)-1).bit_length()
    record_bytes=8+8  # Maximum native key width plus original int64 index.
    array_bytes=padded*record_bytes
    scratch_bytes=2*array_bytes
    if scratch_bytes>sim.output_budget():
        raise ValueError('Consumer sort scratch exceeds one finite output owner')
    sim.stats['consumer_sort_peak_scratch_bytes']=max(sim.stats['consumer_sort_peak_scratch_bytes'],scratch_bytes)
    sim.stats['consumer_sort_keys']+=count
    lanes=sim.c['vector_lanes']
    def vector(operations):
        nonlocal ready
        cost=math.ceil(operations/lanes)
        ready=sim.issue_vector(ready,cost)
        sim.stats['vector_cycles']+=cost
        sim.stats['consumer_sort_vector_cycles']+=cost
    # Initialize stable indices and invalid padded sentinels.
    vector(2*padded)
    base=sim.accumulator_base(4)
    ready=sim.output_transfer(array_bytes,ready,write=True,offset=base)
    key_words=math.ceil(event['inputs'][0]['element_bytes']/4)
    index_words=2
    # Less/equal comparisons of words; tie mask, direction, two selections.
    operations_per_record=2*(key_words+index_words)+5
    source=0;destination=array_bytes
    span=2
    while span<=padded:
        partner=span//2
        while partner:
            ready=sim.output_transfer(array_bytes,ready,offset=base+source)
            ready=sim.output_transfer(array_bytes,ready,offset=base+source)
            vector(padded*operations_per_record)
            ready=sim.output_transfer(array_bytes,ready,write=True,offset=base+destination)
            sim.stats['consumer_sort_compare_exchange_layers']+=1
            source,destination=destination,source
            partner//=2
        span*=2
    # Extract the int64 permutation through the same banked owner.
    ready=sim.output_transfer(array_bytes,ready,offset=base+source)
    vector(count)
    return ready


def operand_capacity(config,element_bytes):
    slots=config['arrays']*config.get('operand_buffers_per_array',2)
    # FP16 and FP32 share physical owners. Narrow payloads cannot move their
    # slot bases while another precision still has an in-flight consumer.
    storage_bytes=max(np.dtype(np.float32).itemsize,element_bytes)
    size=(config['array_rows']+config['array_cols'])*config['k_tile']*storage_bytes
    reserved=slots*size
    if config.get('stream_operand_slices',False):
        bank_row=config['sram_banks']*SECTOR_BYTES
        if any(span%bank_row for span in (config['array_rows']*config['k_tile']*storage_bytes,
                config['array_cols']*config['k_tile']*storage_bytes,input_latch_bytes(config))):
            raise ValueError('Streamed operand and accumulator regions require complete physical bank rows')
    latch_charge=input_latch_bytes(config) if config.get('input_latch_budget','output')=='output' else 0
    remaining=config['output_sram_bytes']-reserved-latch_charge
    if remaining<=0:raise ValueError('Operand slots exhaust output SRAM')
    return slots,size,remaining//config['output_buffers']


@core.njit(cache=True)
def rectangle(starts,width,stride,element_bytes=1):
    """Exact byte-enabled sectors for a strided matrix tile."""
    if not len(starts) or not width:
        return np.empty(0,np.int64),np.empty(0,np.uint64)
    contiguous=stride<=8
    count=(width*stride+SECTOR_BYTES-1)//SECTOR_BYTES+1 if contiguous else width*((element_bytes+2*SECTOR_BYTES-2)//SECTOR_BYTES)
    sectors=np.empty(len(starts)*count,np.int64);used=0
    for start in starts:
        if contiguous:
            first=start//SECTOR_BYTES;last=(start+(width-1)*stride+element_bytes-1)//SECTOR_BYTES
            for sector in range(first,last+1):sectors[used]=sector;used+=1
        else:
            for column in range(width):
                first=(start+column*stride)//SECTOR_BYTES
                last=(start+column*stride+element_bytes-1)//SECTOR_BYTES
                for sector in range(first,last+1):sectors[used]=sector;used+=1
    sectors=sectors[:used]
    tags=np.unique(sectors//SLOT_SECTORS)
    masks=np.zeros(len(tags),np.uint64)
    for sector in sectors:
        where=np.searchsorted(tags,sector//SLOT_SECTORS)
        masks[where]|=np.uint64(1)<<np.uint64(sector%SLOT_SECTORS)
    return tags,masks


@core.njit(cache=True)
def accumulator(targets,column,width,output_width,earliest,state,banks,calendar,lengths,bank_width,counts,write,base=0):
    """32-byte interleaving of the separate output window over bank ports."""
    demand=np.zeros(len(banks),np.int64)
    for target in targets:
        first=(base+(target*output_width+column)*4)//SECTOR_BYTES
        last=(base+(target*output_width+column+width)*4-1)//SECTOR_BYTES
        for sector in range(first,last+1):demand[sector%len(banks)]+=SECTOR_BYTES
    done=earliest
    for bank in range(len(banks)):
        if not demand[bank]:continue
        end=core.reserve_bank(bank,earliest,(demand[bank]+bank_width-1)//bank_width,state[0],calendar,lengths)
        banks[bank]=max(banks[bank],end);done=max(done,end)
        counts[7 if write else 6]+=demand[bank]
    return done


@core.njit(cache=True)
def window_transfer(offset,nbytes,earliest,state,banks,calendar,lengths,bank_width,counts,write):
    demand=np.zeros(len(banks),np.int64)
    for sector in range(offset//SECTOR_BYTES,(offset+nbytes+SECTOR_BYTES-1)//SECTOR_BYTES):
        demand[sector%len(banks)]+=SECTOR_BYTES
    done=earliest
    for bank in range(len(banks)):
        if not demand[bank]:continue
        end=core.reserve_bank(bank,earliest,(demand[bank]+bank_width-1)//bank_width,state[0],calendar,lengths)
        banks[bank]=max(banks[bank],end);done=max(done,end)
        counts[7 if write else 6]+=demand[bank]
    return done


@core.njit(cache=True)
def operand_transfer(base,nrows,column,depth,pitch,element_bytes,earliest,state,
                     banks,calendar,lengths,bank_width,counts,write):
    """Actual fixed-pitch operand rows in a reversible skewed bank layout."""
    demand=np.zeros(len(banks),np.int64)
    for row in range(nrows):
        first=(base+row*pitch+column*element_bytes)//SECTOR_BYTES
        last=(base+row*pitch+(column+depth)*element_bytes-1)//SECTOR_BYTES
        for address in range(first,last+1):
            bank=(address//len(banks)+address)%len(banks)
            demand[bank]+=SECTOR_BYTES
    done=earliest
    for bank in range(len(banks)):
        if not demand[bank]:continue
        end=core.reserve_bank(bank,earliest,(demand[bank]+bank_width-1)//bank_width,
                              state[0],calendar,lengths)
        banks[bank]=max(banks[bank],end);done=max(done,end)
        counts[7 if write else 6]+=demand[bank]
    return done


@core.njit(cache=True)
def stream_inputs(a_base,b_base,nrows,width,depth,pitch,element_bytes,precision,
                  unit,earliest,latches,state,banks,calendar,lengths,bank_width,
                  counts,activity,activity_at):
    """Read the next slice while a previous input-register group feeds K.

    A tile keeps one array throughout any data gap. The original array input
    skew registers consume the lane words; the final systolic tail is paid by
    the caller once, without a partial-sum round trip at each slice boundary.
    """
    quantum=SECTOR_BYTES//element_bytes;issued=earliest;occupied=0;records=0
    for slice_at,column in enumerate(range(0,depth,quantum)):
        count=min(quantum,depth-column);owner=slice_at%2
        available=max(earliest,latches[unit,owner])
        a=operand_transfer(a_base,nrows,column,count,pitch,element_bytes,available,
            state,banks,calendar,lengths,bank_width,counts,False)
        b=operand_transfer(b_base,width,column,count,pitch,element_bytes,available,
            state,banks,calendar,lengths,bank_width,counts,False)
        # Registered bank-to-lane routing. No array may consume an unfilled
        # lane, and a register group cannot be overwritten until consumption.
        begin=max(issued,max(a,b)+1);end=begin+count*precision
        latches[unit,owner]=end;issued=end;occupied+=count*precision
        if activity is not None:
            if records and activity[activity_at+records-1,1]==begin:
                activity[activity_at+records-1,1]=end
            else:
                activity[activity_at+records,0]=begin
                activity[activity_at+records,1]=end
                activity[activity_at+records,2]=unit;records+=1
    return issued,occupied,records


@core.njit(cache=True)
def stream_batch(programs,targets,columns,precision,bm,bn,bk,element_bytes,
                 feeder,feeder_b,arrays,latches,retain,state,banks,calendar,
                 lengths,bank_width,counts,partial,accumulator_base,
                 activity,activity_at):
    """Advance a closed set of at most one current microprogram per array.

    Only the next slice competes for the existing ports. No other producer can
    submit a request during this call; later admission must use the resulting
    dispatch floor. This makes calendar retirement safe without larger queues.
    Completions retain actual operand/partial owners until their last consumer.
    """
    quantum=SECTOR_BYTES//element_bytes;occupied=0;records=0;peak=0
    active=np.full(len(arrays),-1,np.int64);completion=state[0]
    for at in range(len(programs)):
        p=programs[at]
        if not p[S_VALID]:continue
        if retain:
            unit=p[S_UNIT]
            if active[unit]>=0:raise AssertionError('Two retained owners on one array')
            active[unit]=at;p[S_ISSUED]=max(p[S_READY],arrays[unit])
        else:p[S_UNIT]=-1
    while True:
        # Prefetched commands keep their operand halves, but only the current
        # command owns array/latch state. No read from the next command occurs
        # until the original array issue stream and PE owner allow it.
        if not retain:
            for _ in range(len(arrays)):
                unit=-1
                for candidate in range(len(arrays)):
                    if active[candidate]>=0:continue
                    if unit<0 or arrays[candidate]<arrays[unit]:unit=candidate
                if unit<0:break
                at=-1
                for candidate in range(len(programs)):
                    p=programs[candidate]
                    if not p[S_VALID] or p[S_UNIT]>=0:continue
                    if at<0 or p[S_READY]<programs[at,S_READY]:at=candidate
                if at<0:break
                p=programs[at];active[unit]=at;p[S_UNIT]=unit
                p[S_ISSUED]=max(p[S_READY],arrays[unit])
        selected=-1;floor=np.iinfo(np.int64).max
        for unit in range(len(arrays)):
            if active[unit]<0:continue
            p=programs[active[unit]]
            group=(p[S_CURSOR]//quantum)%2
            ready=max(state[0],p[S_ISSUED] if p[S_CURSOR]==0 else p[S_READY],latches[unit,group])
            if ready<floor:floor=ready;selected=unit
        if selected<0:break
        unit=selected;p=programs[active[unit]];column=p[S_CURSOR]
        depth=min(quantum,p[S_K]-column);group=(column//quantum)%2
        state[0]=floor
        a=operand_transfer(p[S_A],p[S_M],column,depth,bk*4,element_bytes,floor,
            state,banks,calendar,lengths,bank_width,counts,False)
        b=operand_transfer(p[S_B],p[S_N],column,depth,bk*4,element_bytes,floor,
            state,banks,calendar,lengths,bank_width,counts,False)
        begin=max(p[S_ISSUED],max(a,b)+1);end=begin+depth*precision
        p[S_ISSUED]=end;p[S_CURSOR]+=depth;latches[unit,group]=end
        occupied+=depth*precision
        if activity is not None:
            activity[activity_at+records,0]=begin
            activity[activity_at+records,1]=end
            activity[activity_at+records,2]=unit;records+=1
        state[0]=floor+1
        if p[S_CURSOR]==p[S_K]:
            consumed=p[S_ISSUED];done=consumed+bm+bn-2
            arrays[unit]=consumed
            feeder[p[S_ASLOT]]=max(feeder[p[S_ASLOT]],consumed)
            feeder_b[p[S_BSLOT]]=max(feeder_b[p[S_BSLOT]],consumed)
            if p[S_COMMIT]:
                target=targets[p[S_ROW]:p[S_END]]
                done=accumulator(target,p[S_COLUMN],p[S_N],columns,done,state,
                    banks,calendar,lengths,bank_width,counts,True,accumulator_base)
                for t in target:partial[t,p[S_COLUMN]//bn]=done
                if retain:arrays[unit]=max(arrays[unit],done)
            completion=max(completion,done);p[S_VALID]=0;active[unit]=-1
        peak=max(peak,np.max(lengths))
    return completion,occupied,records,peak



@core.njit(cache=True)
def drain_output(offset,nbytes,earliest,state,pending,banks,calendar,lengths,
                 bank_width,counts,burst_cycles,latency,activity=None):
    """Commit a bounded output window through its real SRAM/DMA ports.

    Each burst is written before being read by DMA. No feature-directory
    copy is needed for an externally visible result with no later consumer.
    The caller retains the output window until the last DMA return.
    """
    done=earliest
    for begin in range(0,nbytes,DMA_BURST_BYTES):
        size=min(DMA_BURST_BYTES,nbytes-begin)
        stored=window_transfer(offset+begin,size,earliest,state,banks,calendar,lengths,bank_width,counts,True)
        loaded=window_transfer(offset+begin,size,stored,state,banks,calendar,lengths,bank_width,counts,False)
        returned=core.transaction(loaded,True,state,pending,counts,burst_cycles,latency,DMA_BURST_BYTES,activity)
        done=max(done,returned)
    return done


def inverse_row_scratch(count,source_rows):
    """Input keys, two scan vectors, mutable cursors and consumer IDs."""
    align=lambda size:(size+SECTOR_BYTES-1)//SECTOR_BYTES*SECTOR_BYTES
    keys=align(count*8);scan=align((source_rows+1)*4)
    return keys,keys+scan,keys+2*scan,keys+3*scan,keys+3*scan+align(count*4)


def unique_row_scratch(count,source_rows):
    """Keys, two presence/rank scan banks, then inverse row ordinals."""
    align=lambda size:(size+SECTOR_BYTES-1)//SECTOR_BYTES*SECTOR_BYTES
    keys=align(count*8);scan=align(source_rows*4)
    return keys,keys+scan,keys+2*scan,keys+2*scan+align(count*4)


@core.njit(cache=True)
def presence_rank_transfer(mapping,n,layout,earliest,lanes,state,banks,calendar,lengths,bank_width,counts,scratch_base=0):
    """Unique/inverse without constructing an unused CSR consumer list.

    Presence stores use ordinary bank arbitration, one pending key per bank
    per round, even for duplicate keys. No atomic histogram, free bitmap
    update, full native-map registers or extra scatter ports are assumed.
    """
    a,b,inverse_base,_=layout
    a+=scratch_base;b+=scratch_base;inverse_base+=scratch_base
    count=len(mapping);flags=np.zeros(n,np.int64);operations=0
    for row in mapping:
        if row<0 or row>=n:raise ValueError('Unique native row outside source domain')
    ready=earliest+(3*count+lanes-1)//lanes
    operations+=(3*count+lanes-1)//lanes
    ready=window_transfer(scratch_base,count*8,ready,state,banks,calendar,lengths,bank_width,counts,True)
    cost=(n+lanes-1)//lanes;ready+=cost;operations+=cost
    ready=window_transfer(a,n*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    for begin in range(0,count,lanes):
        end=min(count,begin+lanes)
        ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,False)
        ready+=3;operations+=3
        pending_lane=np.ones(end-begin,np.bool_);remaining=end-begin
        while remaining:
            used_banks=np.zeros(len(banks),np.bool_);written=ready
            for local in range(end-begin):
                if not pending_lane[local]:continue
                row=mapping[begin+local]
                bank=((a+row*4)//SECTOR_BYTES)%len(banks)
                if used_banks[bank]:continue
                used_banks[bank]=True;pending_lane[local]=False;remaining-=1
                flags[row]=1
                written=max(written,window_transfer(a+row*4,4,ready,state,banks,calendar,lengths,bank_width,counts,True))
            ready=written+1;operations+=1
    rank=np.cumsum(flags);used=np.flatnonzero(flags)
    shift=1;src=a;dst=b
    while shift<n:
        ready=window_transfer(src,n*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
        ready=window_transfer(src,(n-shift)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
        cost=(n+lanes-1)//lanes;ready+=cost;operations+=cost
        ready=window_transfer(dst,n*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
        src,dst=dst,src;shift*=2
    # The dead scan bank holds the compact unique list. Stream source-ID
    # chunks; adjacent rank differences determine actual output positions.
    for begin in range(0,n,lanes):
        end=min(n,begin+lanes)
        ready=window_transfer(src+begin*4,(end-begin)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
        previous=max(0,begin-1)
        ready=window_transfer(src+previous*4,(end-previous)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
        cost=(3*(end-begin)+lanes-1)//lanes;ready+=cost;operations+=cost
        emitted=int(flags[begin:end].sum())
        # Rank supplies addresses, not a free compacted data stream. Route
        # the native-ID lanes through the same one-store-per-bank arbiter.
        pending_lane=flags[begin:end]!=0;remaining=emitted
        while remaining:
            used_banks=np.zeros(len(banks),np.bool_);written=ready
            for local in range(end-begin):
                if not pending_lane[local]:continue
                row=begin+local;address=dst+(rank[row]-1)*4
                bank=(address//SECTOR_BYTES)%len(banks)
                if used_banks[bank]:continue
                used_banks[bank]=True;pending_lane[local]=False;remaining-=1
                written=max(written,window_transfer(address,4,ready,state,banks,calendar,lengths,bank_width,counts,True))
            ready=written+1;operations+=1
    inverse=np.empty(count,np.int64)
    for begin in range(0,count,lanes):
        end=min(count,begin+lanes)
        ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,False)
        loaded=ready
        for i in range(begin,end):
            loaded=max(loaded,window_transfer(src+mapping[i]*4,4,ready,state,banks,calendar,lengths,bank_width,counts,False))
            inverse[i]=rank[mapping[i]]-1
        ready=loaded+2;operations+=2
        ready=window_transfer(inverse_base+begin*4,(end-begin)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    ready=window_transfer(dst,len(used)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    ready=window_transfer(inverse_base,count*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    return ready,operations,used,inverse


@core.njit(cache=True)
def identity_row_transfer(mapping,n,layout,earliest,lanes,state,banks,calendar,lengths,bank_width,counts,consumer_ids,scratch_base=0):
    """Validate a complete one-to-one dispatch, then emit its ordinary CSR.

    Only one input-lane packet is compared at a time. A running scalar AND
    carries the proof across packets; no invocation-sized register map is
    retained. The original key scratch holds both words of each request.
    Failure returns after the probe, before constructing any output CSR.
    """
    count=len(mapping)
    if count!=n or len(consumer_ids)!=count:raise ValueError('Identity dispatch extent mismatch')
    ready=earliest;operations=0;valid=True
    # The caller has read the raw signed index payload through the hierarchy.
    # Normalize and stage each packet; do not inherit the host's normalized
    # reference CSR as a free runtime proof. Keep real global-consumer words.
    for begin in range(0,count,lanes):
        end=min(count,begin+lanes)
        ready+=3;operations+=3 # sign compare, conditional add, select
        ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,True)
        ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,False)
        # iota/add, compare, bounded AND tree, scalar accumulation.
        depth=0;width=1
        while width<end-begin:width*=2;depth+=1
        cost=3+depth;ready+=cost;operations+=cost
        for row in range(begin,end):
            native=mapping[row]
            if native<0:native+=n
            valid=valid and native==row
    ready+=1;operations+=1 # branch after the final proof packet
    if not valid:return False,ready,operations,np.empty(0,np.int64),np.empty(0,np.int64)
    hist_a,_,_,row_base,_=layout
    hist_a+=scratch_base;row_base+=scratch_base
    for begin in range(0,n+1,lanes):
        size=min(lanes,n+1-begin)*4
        ready+=1;operations+=1 # iota/add for offsets
        ready=window_transfer(hist_a+begin*4,size,ready,state,banks,calendar,lengths,bank_width,counts,True)
    for begin in range(0,count,lanes):
        end=min(count,begin+lanes)
        # Read the real (representative,global-consumer) pairs again. The
        # identity proof authorizes a positional copy, not synthetic IDs.
        ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,False)
        ready+=1;operations+=1
        ready=window_transfer(row_base+begin*4,(end-begin)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    ready=window_transfer(hist_a,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    ready=window_transfer(row_base,count*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    return True,ready,operations,np.arange(n+1),consumer_ids.copy()


@core.njit(cache=True)
def inverse_row_transfer(mapping,n,layout,earliest,lanes,state,banks,calendar,lengths,bank_width,counts,consumer_ids=None,scratch_base=0):
    """CSR construction with real counter conflicts, scan and scatter ports.

    One vector of keys is live at a time. Existing bank arbiters admit one
    read/modify/write counter per bank per round, then the vector controller
    tests its remaining lane mask. No atomic or associative histogram unit.
    """
    hist_a,hist_b,cursor_base,row_base,_=layout
    hist_a+=scratch_base;hist_b+=scratch_base;cursor_base+=scratch_base;row_base+=scratch_base
    count=len(mapping);values=np.zeros(n+1,np.int64);operations=0
    constructed=np.full(count,-1,np.int64);offsets=np.empty(n+1,np.int64)
    ready=earliest+(3*count+lanes-1)//lanes # sign compare/add/select of native indices
    operations+=(3*count+lanes-1)//lanes
    ready=window_transfer(scratch_base,count*8,ready,state,banks,calendar,lengths,bank_width,counts,True)
    ready+=(n+1+lanes-1)//lanes;operations+=(n+1+lanes-1)//lanes
    ready=window_transfer(hist_a,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    for phase in range(2):
        if phase==1:
            offsets=np.cumsum(values)
            # Inclusive scan of [0,count(row0),...] yields CSR offsets.
            shift=1;src=hist_a;dst=hist_b
            while shift<n+1:
                ready=window_transfer(src,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
                ready=window_transfer(src,(n+1-shift)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
                cost=(n+1-shift+lanes-1)//lanes+(shift+lanes-1)//lanes
                ready+=cost;operations+=cost
                ready=window_transfer(dst,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
                src,dst=dst,src;shift*=2
            hist_a=src
            ready=window_transfer(hist_a,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
            ready=window_transfer(cursor_base,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
            values=offsets.copy()
        for begin in range(0,count,lanes):
            end=min(count,begin+lanes)
            ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,False)
            # Counter row, byte address, bank decode are vector operations.
            ready+=3;operations+=3
            pending_lane=np.ones(end-begin,np.bool_)
            remaining=end-begin
            while remaining:
                selected=np.empty(len(banks),np.int64);selected_count=0
                used=np.zeros(len(banks),np.bool_)
                base=hist_a if phase==0 else cursor_base
                for local in range(end-begin):
                    if not pending_lane[local]:continue
                    row=mapping[begin+local]+(1 if phase==0 else 0)
                    bank=((base+row*4)//SECTOR_BYTES)%len(banks)
                    if used[bank]:continue
                    used[bank]=True;pending_lane[local]=False;remaining-=1
                    selected[selected_count]=begin+local;selected_count+=1
                loaded=ready
                for item in range(selected_count):
                    row=mapping[selected[item]]+(1 if phase==0 else 0)
                    loaded=max(loaded,window_transfer(base+row*4,4,ready,state,banks,calendar,lengths,bank_width,counts,False))
                added=loaded+1;written=added;operations+=1
                if phase==1:
                    # Cursor-dependent scatter address goes through the
                    # existing integer vector/AGU path before SRAM issue.
                    added+=1;written=added;operations+=1
                for item in range(selected_count):
                    i=selected[item];row=mapping[i]+(1 if phase==0 else 0)
                    if phase==1:
                        position=values[row]
                        constructed[position]=i if consumer_ids is None else consumer_ids[i]
                        written=max(written,window_transfer(row_base+position*4,4,added,state,banks,calendar,lengths,bank_width,counts,True))
                    values[row]+=1
                    written=max(written,window_transfer(base+row*4,4,added,state,banks,calendar,lengths,bank_width,counts,True))
                ready=written+1;operations+=1 # pending-lane mask test
    ready=window_transfer(hist_a,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    ready=window_transfer(row_base,count*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    return ready,operations,offsets,constructed


@core.njit(cache=True)
def unique_row_transfer(mapping,n,layout,earliest,lanes,state,banks,calendar,lengths,bank_width,counts,scratch_base=0):
    """Reuse bounded CSR construction, then rank native IDs and emit inverse.

    Scratch is inverse_row_scratch: no full map lives in registers. The
    counting scan has already paid for conflict arbitration and CSR scatter.
    Its dead cursor/consumer storage now holds ranks and inverse ordinals.
    """
    ready,operations,offsets,order=inverse_row_transfer(mapping,n,layout,earliest,lanes,
        state,banks,calendar,lengths,bank_width,counts,scratch_base=scratch_base)
    a,b,rank_base,inverse_base,_=layout
    a+=scratch_base;b+=scratch_base;rank_base+=scratch_base;inverse_base+=scratch_base
    scan_passes=0;shift=1
    while shift<n+1:scan_passes+=1;shift*=2
    offsets_base=a if scan_passes%2==0 else b
    scan_tmp=b if offsets_base==a else a
    used=np.flatnonzero(offsets[1:]!=offsets[:-1])
    rank=np.full(n,-1,np.int64);rank[used]=np.arange(len(used))
    # Build nonempty flags from adjacent offsets, then a finite prefix scan.
    ready=window_transfer(offsets_base,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    cost=(2*n+lanes-1)//lanes;ready+=cost;operations+=cost
    ready=window_transfer(rank_base,n*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    shift=1;src=rank_base;dst=scan_tmp
    while shift<n:
        ready=window_transfer(src,n*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
        ready=window_transfer(src,(n-shift)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
        cost=(n+lanes-1)//lanes;ready+=cost;operations+=cost
        ready=window_transfer(dst,n*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
        src,dst=dst,src;shift*=2
    # Scan ranks are read through actual native addresses. Inverse ordinals
    # retain original leaf order; output stores therefore need no free sort.
    inverse=np.empty(len(mapping),np.int64)
    for begin in range(0,len(mapping),lanes):
        end=min(len(mapping),begin+lanes)
        ready=window_transfer(scratch_base+begin*8,(end-begin)*8,ready,state,banks,calendar,lengths,bank_width,counts,False)
        loaded=ready
        for i in range(begin,end):
            loaded=max(loaded,window_transfer(src+mapping[i]*4,4,ready,state,banks,calendar,lengths,bank_width,counts,False))
            inverse[i]=rank[mapping[i]]
        ready=loaded+2;operations+=2
        ready=window_transfer(inverse_base+begin*4,(end-begin)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    # Native IDs in ascending order become the unique gather list. Read the
    # original offsets and final ranks before compacting, then drain both
    # lists to the ordinary feature/index store under this same output owner.
    ready=window_transfer(offsets_base,(n+1)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    ready=window_transfer(src,n*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    cost=(3*n+lanes-1)//lanes;ready+=cost;operations+=cost
    unique_base=dst
    ready=window_transfer(unique_base,len(used)*4,ready,state,banks,calendar,lengths,bank_width,counts,True)
    ready=window_transfer(unique_base,len(used)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    ready=window_transfer(inverse_base,len(mapping)*4,ready,state,banks,calendar,lengths,bank_width,counts,False)
    return ready,operations,used,inverse


@core.njit(cache=True)
def grouped_product(at,mtiles,ntiles,kt,chunks):
    """K-block -> M -> K-within-block -> N, including a short last block."""
    start=at//(mtiles*ntiles*chunks)*chunks
    local=at%(mtiles*ntiles*chunks)
    length=min(chunks,kt-start)
    return local//(length*ntiles),local%ntiles,start+local//ntiles%length


@core.njit(cache=True)
def replace_weight_slot(owners,ready,offset,k,bk,chunks):
    """Retain this admitted K block until its M consumers have completed."""
    first=k//bk//chunks*chunks*bk
    limit=first+chunks*bk
    slot=-1
    for candidate in range(len(ready)):
        current=(owners[candidate,0]==offset and
                 first<=owners[candidate,2]<limit)
        if not current and (slot<0 or ready[candidate]<ready[slot]):
            slot=candidate
    if slot<0:raise AssertionError('A fitting weight set has no replaceable half-slot')
    return slot


@core.njit(cache=True)
def _copy_profile_add(profile,field,amount):
    # A separate optional-argument specialization also supports observers
    # disabled in a source tuple; this does not affect controller timing.
    if profile is not None:profile[field]+=amount


@core.njit(cache=True)
def paired_copy(a,b,state,pending,priorities,victim_scan,counts,lookup_width,
                cam_latency,bank_width,burst_cycles,dram_latency,independent_drain,
                separate_read_completion,dma_activity,batch,control_registers,queue,
                ready_first,source_credit,source_defer,warm=False,first_b=False,shared_directory=False,
                mask_pipeline=False):
    """One finite descriptor program for two bounded operand destinations.

    Both sources share CAM issue, the circular descriptor queue, route
    registers, output banks and dispatch frontier. Only the mask/controller
    traversal is combined; every byte still traverses copy_transfer().
    """
    if queue is None:raise ValueError('Paired copy requires the original descriptor queue')
    if ready_first or source_credit or source_defer is not None or not control_registers:
        raise ValueError('Paired copy uses the original registered mask controller')
    chunk=min(batch,(len(queue)-1)//2,len(a[3])-victim_scan-1,len(b[3])-victim_scan-1)
    if shared_directory:chunk=min(chunk,(len(a[3])-victim_scan-1)//2)
    if chunk<2:raise ValueError('Paired copy requires two safe source descriptors')
    complete=np.full(2,state[0],np.int64);warmed=0
    for begin in range(0,max(len(a[0]),len(b[0])),chunk):
        sizes=np.array([min(chunk,max(0,len(a[0])-begin)),min(chunk,max(0,len(b[0])-begin))],np.int64)
        size=sizes.sum();owners=np.empty((2,chunk),np.int64)
        # The same original fill path admits both directory streams. These
        # references fit the combined queue and each directory's victim guard.
        for side in range(2):
            operand=(int(first_b)+side)%2
            source=a if operand==0 else b
            tags,masks,directory,slots,links,meta,banks,calendar,lengths,layout,profile,sink,credit=source
            count=sizes[operand]
            if not count:continue
            warmed+=int(begin==0 and warm and state[0]<credit)
            if not warm:state[0]=max(state[0],credit)
            core.access_base(tags[begin:begin+count],masks[begin:begin+count],directory,slots,links,meta,
                state,pending,banks,calendar,lengths,priorities,victim_scan,counts,False,0,
                lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,SECTOR_BYTES,
                independent_drain,layout,separate_read_completion,profile,dma_activity,True,None)
        capacity=len(queue)-1;cursor=queue[-1]
        for at in range(size):state[0]=max(state[0],queue[(cursor+at)%capacity])
        state[0]+=(size+lookup_width-1)//lookup_width
        state[0]+=(size+lookup_width-1)//lookup_width+cam_latency
        state[0]+=(size*8+lookup_width-1)//lookup_width
        enabled=np.uint64(0)
        for operand in range(2):
            source=a if operand==0 else b
            tags,masks,directory,slots,links,meta,banks,calendar,lengths,layout,profile,sink,credit=source
            view=slots.view(np.uint64)
            for at in range(sizes[operand]):
                tag=tags[begin+at]
                if tag not in directory:raise AssertionError('Paired copy owner was evicted')
                slot=directory[tag];owners[operand,at]=slot
                if slots[slot,0]!=tag or masks[begin+at]&~view[slot,1]:
                    raise AssertionError('Paired copy owner has incomplete fill')
                enabled|=masks[begin+at]
            counts[3]+=sizes[operand]
            _copy_profile_add(profile,3,sizes[operand])
        levels=0;leaves=1
        while leaves<size:leaves*=2;levels+=1
        state[0]+=levels
        if mask_pipeline and enabled:state[0]+=1
        returns=np.full((2,chunk),state[0],np.int64)
        for bit in range(SLOT_SECTORS):
            if not enabled&(np.uint64(1)<<np.uint64(bit)):continue
            state[0]+=(0 if mask_pipeline else 1)+(size+lookup_width-1)//lookup_width
            for side in range(2):
                operand=(int(first_b)+side)%2
                source=a if operand==0 else b
                tags,masks,directory,slots,links,meta,banks,calendar,lengths,layout,profile,sink,credit=source
                for at in range(sizes[operand]):
                    mask=masks[begin+at]&(np.uint64(1)<<np.uint64(bit))
                    if not mask:continue
                    slot=owners[operand,at];tag=tags[begin+at]
                    ready=max(state[0],slots[slot,5],credit)
                    read,end,nread,nwrite=core.copy_transfer(slot,tag,mask,ready,state,banks,
                        calendar,lengths,bank_width,layout,sink)
                    slots[slot,3]=max(slots[slot,3],read)
                    returns[operand,at]=max(returns[operand,at],read)
                    counts[6]+=nread;counts[7]+=nwrite
                    _copy_profile_add(profile,6,nread)
                    complete[operand]=max(complete[operand],end)
        for operand in range(2):
            offset=0 if operand==0 else sizes[0]
            for at in range(sizes[operand]):queue[(cursor+offset+at)%capacity]=returns[operand,at]
        queue[-1]=(cursor+size)%capacity;state[0]+=1
        if warm:state[0]=max(state[0],a[-1] if sizes[0] else 0,b[-1] if sizes[1] else 0)
    return complete,warmed


@core.njit(cache=True)
def products(source_starts,targets,offset_ptr,weight_bases,weight_k_stride,weight_n_stride,
             source_stride,inner,columns,output_rows,element_bytes,precision,bm,bn,bk,
             feeder,arrays,wave_pipeline,
             feature_directory,weight_directory,feature_slots,weight_slots,
             feature_links,weight_links,feature_meta,weight_meta,
             state,pending,banks,calendar,lengths,priorities,victim_scan,counts,
             lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,independent_drain,
             reuse,feeder_b,k_outer,striped,keep_accumulator,separate_read_completion=False,
             feature_ports=None,weight_ports=None,accumulator_base=0,ready_order=False,activity=None,
             feature_profile=None,weight_profile=None,dma_activity=None,
             stream=False,latches=None,interleave=False,prefetch=1,warm_operands=False,
             bounded_copy=False,copy_ready=None,weight_striped=None,group_k_outer=False,block_reduction=False,
             retain_weight_set=False,packed_operands=False,physical_bk=0,zero_seed=False,copy_batch=0,weight_copy_batch=-1,
             copy_control_registers=False,copy_queue=None,copy_ready_first=False,copy_source_credit=False,
             copy_source_defer=None,operand_skew=False,dual_supply=False,adjacent_supply=False,
             copy_mask_pipeline=False,shared_directory=False):
    fb,fc,fl=(banks,calendar,lengths) if feature_ports is None else feature_ports
    wb,wc,wl=(banks,calendar,lengths) if weight_ports is None else weight_ports
    weight_layout=striped if weight_striped is None else weight_striped
    if weight_copy_batch<0:weight_copy_batch=copy_batch
    partial=np.zeros((output_rows,(columns+bn-1)//bn),np.int64)
    # Only a compiler-proven dense, one-contribution-per-row first K can
    # seed +0 locally. Sparse offsets and streaming keep their original path.
    zero_seed=zero_seed and inner>0 and len(offset_ptr)==2 and not stream
    initial=state[0]
    if not zero_seed:
        initial=accumulator(np.arange(output_rows),0,columns,columns,state[0],
            state,banks,calendar,lengths,bank_width,counts,True,accumulator_base)
    partial[:,:]=initial
    completion=initial;tiles=0;occupied=0;payload_max=0;reused_bytes=0;records=0
    programs=np.zeros((len(arrays)*prefetch,STREAM_DESCRIPTOR_WORDS),np.int64)
    batch_count=0;peak_calendar=0;batches=0;warm_requests=0;adjacent_batches=0;adjacent_bytes=0
    # Packed narrow K is increased by exactly the inverse element ratio.
    # bk*storage_bytes and every physical half-slot base remain unchanged.
    storage_bytes=element_bytes if packed_operands else max(4,element_bytes)
    slot_pitch=(physical_bk*max(4,element_bytes) if physical_bk else bk*storage_bytes)
    # These tags describe bounded operand registers, not another feature cache.
    # Tags are reset at each output window. Ready retains the final consumer
    # time across windows; fill time is separate so concurrent reads are legal.
    a_owner=np.full((len(feeder),3),-1,np.int64)
    b_owner=np.full((len(feeder_b),3),-1,np.int64)
    a_fill=np.zeros(len(feeder),np.int64)
    b_fill=np.zeros(len(feeder_b),np.int64)
    # Only one partial-sum owner per physical array. In K-outer order this
    # is possible when every N tile in the row group has its own array. Sparse
    # offsets change target packing, so their final results still commit.
    retain=keep_accumulator and inner>bk and (not k_outer or (columns+bn-1)//bn<=len(arrays))
    # K -> M -> N keeps the current weight stripe in the existing B half-slots.
    # Every output still receives K in order, with its existing partial ready.
    # PE-retained and stream descriptors keep their original owner schedule.
    group_k_outer=group_k_outer and reuse and not retain and not stream and not interleave
    for offset in range(len(offset_ptr)-1):
        first=offset_ptr[offset];limit=offset_ptr[offset+1]
        mtiles=(limit-first+bm-1)//bm
        ntiles=(columns+bn-1)//bn;kt=(inner+bk-1)//bk
        # One existing B half-slot for each (N tile,K chunk). No partial-sum
        # owner is added: each current N tile holds exactly one real array.
        chunks=min(kt,len(feeder_b)//ntiles) if block_reduction and group_k_outer and 0<ntiles<=len(arrays) else 1
        blocked=chunks>1
        holding=retain or blocked
        owner_outer=k_outer or blocked
        owners=np.full(ntiles,-1,np.int64)
        for group in range(mtiles):
            for product in range(ntiles*kt):
                if group_k_outer:
                    at=group*ntiles*kt+product
                    mi,ni,ki=grouped_product(at,mtiles,ntiles,kt,chunks)
                    k=ki*bk;row=first+mi*bm;column=ni*bn
                else:
                    row=first+group*bm
                    column=(product%ntiles if k_outer else product//kt)*bn
                    k=(product//ntiles if k_outer else product%kt)*bk
                end_row=min(row+bm,limit)
                target=targets[row:end_row]
                width=min(bn,columns-column)
                depth=min(bk,inner-k)
                first_partial=k==0 or (blocked and k//bk%chunks==0)
                final_partial=k+depth==inner or (blocked and (k//bk+1)%chunks==0)
                if holding and owner_outer and first_partial and column==0:owners[:]=-1
                payload=(end_row-row+width)*depth*element_bytes
                slot_bytes=(bm+bn)*slot_pitch
                a_payload=(end_row-row)*depth*element_bytes
                b_payload=width*depth*element_bytes
                if interleave:
                    conflict=batch_count==(len(arrays) if retain else len(programs))
                    for program in programs:
                        if not program[S_VALID]:continue
                        # A second K product of the same output waits for its
                        # actual commit; retained partials cannot cross M owners.
                        if ((program[S_ROW]==row and program[S_COLUMN]==column)
                            or (retain and program[S_ROW]!=row)):conflict=True
                    if conflict:
                        done,busy,nrecords,peak=stream_batch(programs,targets,columns,precision,bm,bn,bk,
                            element_bytes,feeder,feeder_b,arrays,latches,retain,state,banks,calendar,
                            lengths,bank_width,counts,partial,accumulator_base,activity,records)
                        completion=max(completion,done);occupied+=busy;records+=nrecords
                        peak_calendar=max(peak_calendar,peak);batches+=1;batch_count=0
                a_hit=-1;b_hit=-1
                if reuse:
                    for candidate in range(len(feeder)):
                        if a_owner[candidate,0]==offset and a_owner[candidate,1]==row and a_owner[candidate,2]==k:a_hit=candidate
                    for candidate in range(len(feeder_b)):
                        if b_owner[candidate,0]==offset and b_owner[candidate,1]==column and b_owner[candidate,2]==k:b_hit=candidate
                slot=a_hit if a_hit>=0 else np.argmin(feeder)
                bslot=b_hit if b_hit>=0 else np.argmin(feeder_b)
                if (b_hit<0 and retain_weight_set and group_k_outer and
                        ntiles*chunks<=len(feeder_b)):
                    # The current weights all fit existing physical slots.
                    # Do not replace a current weight merely because its
                    # preceding M consumer completed earlier than an old K.
                    bslot=replace_weight_slot(b_owner,feeder_b,offset,k,bk,chunks)
                if interleave:
                    # A half-slot referenced by an admitted descriptor is
                    # pinned even before its future last-K time is known.
                    if a_hit<0:
                        slot=-1
                        for candidate in range(len(feeder)):
                            pinned=False
                            for program in programs:
                                if program[S_VALID] and program[S_ASLOT]==candidate:pinned=True
                            if not pinned and (slot<0 or feeder[candidate]<feeder[slot]):slot=candidate
                    if b_hit<0:
                        bslot=-1
                        for candidate in range(len(feeder_b)):
                            pinned=False
                            for program in programs:
                                if program[S_VALID] and program[S_BSLOT]==candidate:pinned=True
                            if not pinned and (bslot<0 or feeder_b[candidate]<feeder_b[bslot]):bslot=candidate
                    if slot<0 or bslot<0:raise AssertionError('No unpinned feeder for a bounded array batch')
                paired=(dual_supply and copy_control_registers and not copy_ready_first and not copy_source_credit and
                        copy_source_defer is None and a_hit<0 and b_hit<0 and copy_queue is not None and
                        min(copy_batch,weight_copy_batch,(len(copy_queue)-1)//2,
                            (len(feature_slots)-victim_scan-1)//(2 if shared_directory else 1),
                            len(weight_slots)-victim_scan-1)>=2)
                first_b=ready_order and reuse and b_hit<0 and (a_hit>=0 or feeder_b[bslot]<feeder[slot])
                next_slot=-1
                if (adjacent_supply and dual_supply and group_k_outer and chunks==1 and
                        not holding and ni==0 and a_hit<0 and b_hit>=0 and row+bm<limit and
                        copy_control_registers and not copy_ready_first and not copy_source_credit and
                        copy_source_defer is None and copy_queue is not None and
                        min(copy_batch,(len(copy_queue)-1)//2,(len(feature_slots)-victim_scan-1)//2)>=2):
                    next_hit=False
                    for candidate in range(len(feeder)):
                        if (a_owner[candidate,0]==offset and a_owner[candidate,1]==row+bm and
                                a_owner[candidate,2]==k):next_hit=True
                        if candidate!=slot and feeder[candidate]<=state[0] and (
                                next_slot<0 or feeder[candidate]<feeder[next_slot]):next_slot=candidate
                    if next_hit:next_slot=-1
                adjacent=next_slot>=0
                if adjacent and copy_queue is not None and copy_source_defer is None:
                    a_starts=source_starts[row:end_row]+k*source_stride
                    a_tags,a_masks=rectangle(a_starts,depth,source_stride,element_bytes)
                    a_keys,a_ptr,a_dest=copy_mapping(a_starts,depth,source_stride,element_bytes,
                        slot*bm*slot_pitch,depth*element_bytes)
                    a_sink=(a_keys,a_ptr,a_dest,copy_ready,banks,calendar,lengths,operand_skew)
                    next_end=min(row+2*bm,limit)
                    next_starts=source_starts[row+bm:next_end]+k*source_stride
                    next_tags,next_masks=rectangle(next_starts,depth,source_stride,element_bytes)
                    next_keys,next_ptr,next_dest=copy_mapping(next_starts,depth,source_stride,element_bytes,
                        next_slot*bm*slot_pitch,depth*element_bytes)
                    next_sink=(next_keys,next_ptr,next_dest,copy_ready,banks,calendar,lengths,operand_skew)
                    a=(a_tags,a_masks,feature_directory,feature_slots,feature_links,feature_meta,
                       fb,fc,fl,striped,feature_profile,a_sink,feeder[slot])
                    following=(next_tags,next_masks,feature_directory,feature_slots,feature_links,feature_meta,
                       fb,fc,fl,striped,feature_profile,next_sink,feeder[next_slot])
                    fills,paired_warm=paired_copy(a,following,state,pending,priorities,victim_scan,counts,lookup_width,
                        cam_latency,bank_width,burst_cycles,dram_latency,independent_drain,
                        separate_read_completion,dma_activity,copy_batch,copy_control_registers,copy_queue,
                        copy_ready_first,copy_source_credit,copy_source_defer,warm_operands,False,True,
                        copy_mask_pipeline)
                    a_fill[slot]=fills[0];a_fill[next_slot]=fills[1];warm_requests+=paired_warm
                    a_owner[slot,0]=offset;a_owner[slot,1]=row;a_owner[slot,2]=k
                    a_owner[next_slot,0]=offset;a_owner[next_slot,1]=row+bm;a_owner[next_slot,2]=k
                    # The next M's inevitable first hit is a prepaid copy,
                    # not removed traffic. Cancel exactly that observer hit.
                    prepaid=(next_end-row-bm)*depth*element_bytes
                    reused_bytes-=prepaid;adjacent_batches+=1;adjacent_bytes+=prepaid
                if paired and copy_queue is not None and copy_source_defer is None:
                    a_starts=source_starts[row:end_row]+k*source_stride
                    a_tags,a_masks=rectangle(a_starts,depth,source_stride,element_bytes)
                    a_keys,a_ptr,a_dest=copy_mapping(a_starts,depth,source_stride,element_bytes,
                        slot*bm*slot_pitch,depth*element_bytes)
                    a_sink=(a_keys,a_ptr,a_dest,copy_ready,banks,calendar,lengths,operand_skew)
                    if weight_k_stride<=weight_n_stride:
                        b_starts=weight_bases[offset]+np.arange(column,column+width)*weight_n_stride+k*weight_k_stride
                        b_tags,b_masks=rectangle(b_starts,depth,weight_k_stride,element_bytes)
                    else:
                        b_starts=weight_bases[offset]+np.arange(k,k+depth)*weight_k_stride+column*weight_n_stride
                        b_tags,b_masks=rectangle(b_starts,width,weight_n_stride,element_bytes)
                    b_keys,b_ptr,b_dest=copy_mapping(b_starts,depth if weight_k_stride<=weight_n_stride else width,
                        weight_k_stride if weight_k_stride<=weight_n_stride else weight_n_stride,element_bytes,
                        len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,depth*element_bytes,
                        weight_k_stride>weight_n_stride)
                    b_sink=(b_keys,b_ptr,b_dest,copy_ready,banks,calendar,lengths,operand_skew)
                    # All old consumers must release their half-slot before
                    # the first new write. Warm requests can fill the source
                    # directory earlier, with the normal sink-ready gate.
                    a=(a_tags,a_masks,feature_directory,feature_slots,feature_links,feature_meta,
                       fb,fc,fl,striped,feature_profile,a_sink,feeder[slot])
                    b=(b_tags,b_masks,weight_directory,weight_slots,weight_links,weight_meta,
                       wb,wc,wl,weight_layout,weight_profile,b_sink,feeder_b[bslot])
                    fills,paired_warm=paired_copy(a,b,state,pending,priorities,victim_scan,counts,lookup_width,
                        cam_latency,bank_width,burst_cycles,dram_latency,independent_drain,
                        separate_read_completion,dma_activity,min(copy_batch,weight_copy_batch),
                        copy_control_registers,copy_queue,copy_ready_first,copy_source_credit,copy_source_defer,
                        warm_operands,first_b,shared_directory,copy_mask_pipeline)
                    a_fill[slot]=fills[0];b_fill[bslot]=fills[1];warm_requests+=paired_warm
                    a_owner[slot,0]=offset;a_owner[slot,1]=row;a_owner[slot,2]=k
                    b_owner[bslot,0]=offset;b_owner[bslot,1]=column;b_owner[bslot,2]=k
                for side in range(2):
                    if paired:continue
                    operand=(int(first_b)+side)%2
                    if operand==0:
                        if adjacent:continue
                        if a_hit<0:
                            starts=source_starts[row:end_row]+k*source_stride
                            tags,masks=rectangle(starts,depth,source_stride,element_bytes)
                            warm=warm_operands and reuse and state[0]<feeder[slot]
                            if not warm:state[0]=max(state[0],feeder[slot])
                            sink=None
                            if bounded_copy:
                                keys,ptr,dest=copy_mapping(starts,depth,source_stride,element_bytes,
                                    slot*bm*slot_pitch,slot_pitch if stream else depth*element_bytes)
                                sink=(keys,ptr,dest,copy_ready,banks,calendar,lengths,stream or operand_skew)
                            consumer_ready=feeder[slot] if warm else 0
                            ready=core.access(tags,masks,feature_directory,feature_slots,feature_links,feature_meta,
                                state,pending,fb,fc,fl,priorities,victim_scan,counts,False,consumer_ready,
                                lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,SECTOR_BYTES,independent_drain,striped,separate_read_completion,feature_profile,dma_activity,False,sink,copy_batch,copy_control_registers,copy_queue,copy_ready_first,copy_source_credit,copy_source_defer,copy_mask_pipeline)
                            if warm:state[0]=max(state[0],feeder[slot])
                            warm_requests+=int(warm)
                            if reuse:
                                if bounded_copy:a_fill[slot]=ready
                                elif stream:
                                    a_fill[slot]=operand_transfer(slot*bm*slot_pitch,end_row-row,0,
                                        depth,slot_pitch,element_bytes,ready,state,banks,calendar,lengths,bank_width,counts,True)
                                else:
                                    a_fill[slot]=window_transfer(slot*bm*slot_pitch,a_payload,ready,
                                        state,banks,calendar,lengths,bank_width,counts,True)
                                a_owner[slot,0]=offset;a_owner[slot,1]=row;a_owner[slot,2]=k
                        else:
                            ready=a_fill[slot];reused_bytes+=a_payload
                    else:
                        if b_hit<0:
                            if weight_k_stride<=weight_n_stride:
                                starts=weight_bases[offset]+np.arange(column,column+width)*weight_n_stride+k*weight_k_stride
                                tags,masks=rectangle(starts,depth,weight_k_stride,element_bytes)
                            else:
                                starts=weight_bases[offset]+np.arange(k,k+depth)*weight_k_stride+column*weight_n_stride
                                tags,masks=rectangle(starts,width,weight_n_stride,element_bytes)
                            warm=warm_operands and reuse and state[0]<feeder_b[bslot]
                            if reuse and not warm:state[0]=max(state[0],feeder_b[bslot])
                            sink=None
                            if bounded_copy:
                                keys,ptr,dest=copy_mapping(starts,depth if weight_k_stride<=weight_n_stride else width,
                                    weight_k_stride if weight_k_stride<=weight_n_stride else weight_n_stride,element_bytes,
                                    len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,
                                    slot_pitch if stream else depth*element_bytes,weight_k_stride>weight_n_stride)
                                sink=(keys,ptr,dest,copy_ready,banks,calendar,lengths,stream or operand_skew)
                            consumer_ready=feeder_b[bslot] if warm else 0
                            loaded=core.access(tags,masks,weight_directory,weight_slots,weight_links,weight_meta,
                                state,pending,wb,wc,wl,priorities,victim_scan,counts,False,consumer_ready,
                                lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,SECTOR_BYTES,independent_drain,weight_layout,separate_read_completion,weight_profile,dma_activity,False,sink,weight_copy_batch,copy_control_registers,copy_queue,copy_ready_first,copy_source_credit,copy_source_defer,copy_mask_pipeline)
                            if warm:state[0]=max(state[0],feeder_b[bslot])
                            warm_requests+=int(warm)
                            if reuse:
                                if bounded_copy:b_fill[bslot]=loaded
                                elif stream:
                                    b_fill[bslot]=operand_transfer(len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,
                                        width,0,depth,slot_pitch,element_bytes,loaded,state,banks,calendar,lengths,bank_width,counts,True)
                                else:
                                    b_fill[bslot]=window_transfer(len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,
                                        b_payload,loaded,state,banks,calendar,lengths,bank_width,counts,True)
                                b_owner[bslot,0]=offset;b_owner[bslot,1]=column;b_owner[bslot,2]=k
                            else:ready=max(ready,loaded)
                        else:reused_bytes+=b_payload
                if reuse:ready=max(a_fill[slot],b_fill[bslot],state[0])
                else:ready=window_transfer(slot*slot_bytes,payload,ready,state,banks,calendar,lengths,bank_width,counts,True)
                if not holding or first_partial:
                    for t in target:ready=max(ready,partial[t,column//bn])
                    if not (zero_seed and k==0):
                        ready=accumulator(target,column,width,columns,ready,state,banks,calendar,lengths,bank_width,counts,False,accumulator_base)
                    unit=np.argmin(arrays)
                    if interleave and retain:
                        unit=-1
                        for candidate in range(len(arrays)):
                            if programs[candidate,S_VALID]:continue
                            if unit<0 or arrays[candidate]<arrays[unit]:unit=candidate
                    if holding and owner_outer:
                        # A previously selected array may be ready earlier
                        # than another free array, but it still owns a partial.
                        best=-1
                        for candidate in range(len(arrays)):
                            if candidate in owners:continue
                            if best<0 or arrays[candidate]<arrays[best]:best=candidate
                        unit=best;owners[column//bn]=unit
                elif owner_outer:unit=owners[column//bn]
                begin=max(ready,arrays[unit])
                # The existing PE owner accepts one clear/first-product
                # control beat; subsequent K chunks still read real SRAM.
                if zero_seed and k==0:begin+=1
                if interleave:
                    entry=unit if retain else batch_count
                    if programs[entry,S_VALID]:raise AssertionError('Two microprograms own one descriptor')
                    program=programs[entry]
                    program[S_A]=slot*bm*slot_pitch
                    program[S_B]=len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch
                    program[S_M]=end_row-row;program[S_N]=width;program[S_K]=depth
                    program[S_READY]=begin if retain else ready;program[S_ROW]=row;program[S_END]=end_row
                    program[S_COLUMN]=column;program[S_COMMIT]=not retain or k+depth==inner
                    program[S_ASLOT]=slot;program[S_BSLOT]=bslot
                    program[S_CURSOR]=0;program[S_VALID]=1;program[S_UNIT]=unit if retain else -1
                    state[0]+=1 # fixed descriptor admission
                    batch_count+=1;tiles+=1
                    payload_max=max(payload_max,payload)
                    continue
                if stream:
                    issue_end,busy,nrecords=stream_inputs(slot*bm*slot_pitch,
                        len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,
                        end_row-row,width,depth,slot_pitch,element_bytes,precision,unit,begin,
                        latches,state,banks,calendar,lengths,bank_width,counts,activity,records)
                    records+=nrecords
                    done=issue_end+bm+bn-2
                    arrays[unit]=issue_end if wave_pipeline else done
                elif reuse:
                    if operand_skew:
                        # Nonstream payload is contiguous, including a tail K
                        # whose logical rows may share a physical sector.
                        a_read=operand_transfer(slot*bm*slot_pitch,1,0,(end_row-row)*depth,
                            0,element_bytes,begin,state,banks,calendar,lengths,bank_width,counts,False)
                        b_read=operand_transfer(len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,1,0,width*depth,
                            0,element_bytes,begin,state,banks,calendar,lengths,bank_width,counts,False)
                    else:
                        a_read=window_transfer(slot*bm*slot_pitch,a_payload,begin,
                            state,banks,calendar,lengths,bank_width,counts,False)
                        b_read=window_transfer(len(feeder)*bm*slot_pitch+bslot*bn*slot_pitch,
                            b_payload,begin,state,banks,calendar,lengths,bank_width,counts,False)
                    begin=max(a_read,b_read)
                else:begin=window_transfer(slot*slot_bytes,payload,begin,state,banks,calendar,lengths,bank_width,counts,False)
                duration=depth*precision+bm+bn-2
                if not stream:
                    done=begin+duration
                    arrays[unit]=begin+(depth*precision if wave_pipeline else duration)
                    if activity is not None:
                        activity[records,0]=begin
                        activity[records,1]=arrays[unit]
                        activity[records,2]=unit;records+=1
                # Last K beat has left the feeder. The accumulator is
                # still owned until its delayed bank write completes.
                consumed=issue_end if stream else begin+depth*precision
                feeder[slot]=max(feeder[slot],consumed)
                if reuse:feeder_b[bslot]=max(feeder_b[bslot],consumed)
                if not holding or final_partial:
                    done=accumulator(target,column,width,columns,done,state,banks,calendar,lengths,bank_width,counts,True,accumulator_base)
                    for t in target:partial[t,column//bn]=done
                    if holding:
                        # No second unbudgeted PE accumulator: a new output
                        # owner waits until this tile's final SRAM commit.
                        arrays[unit]=max(arrays[unit],done)
                completion=max(completion,done);tiles+=1
                occupied+=(busy if stream else depth*precision) if wave_pipeline else duration
                payload_max=max(payload_max,(end_row-row+width)*depth*element_bytes)
        if interleave and batch_count:
            # Sparse offsets can touch the same outputs in different packing.
            # Complete their real accumulation before admitting the next offset.
            done,busy,nrecords,peak=stream_batch(programs,targets,columns,precision,bm,bn,bk,
                element_bytes,feeder,feeder_b,arrays,latches,retain,state,banks,calendar,
                lengths,bank_width,counts,partial,accumulator_base,activity,records)
            completion=max(completion,done);occupied+=busy;records+=nrecords
            peak_calendar=max(peak_calendar,peak);batches+=1;batch_count=0
    return completion,tiles,occupied,payload_max,reused_bytes,records,peak_calendar,batches,warm_requests,adjacent_batches,adjacent_bytes


def product_layout(g,index,selected):
    """Address and target lists for one actual row subset, retaining offsets."""
    event=g.events[index];source,weight=event['inputs'][:2]
    if event['kind']=='ancestor_merge':source,weight=event['inputs'][0],event['inputs'][22]
    out=event['outputs'][0];width=out['shape'][-1];inner=source['shape'][-1]
    kind=event['kind']
    offsets=[];targets=[];base=[]
    weight_base=g.bases[(weight['storage'],weight['version'])]+weight['address']-g.storage[weight['storage']]['address']
    if kind=='spconv':
        rule=np.asarray(g.maps['maps'][event['map_key']]['mapping'])
        for offset,row in enumerate(rule):
            valid=row[selected]>=0
            offsets.append(row[selected][valid]);targets.append(np.flatnonzero(valid))
            base.append(weight_base+offset*inner*weight['element_bytes'])
        wk=weight['stride'][-1]*weight['element_bytes'];wn=weight['stride'][0]*weight['element_bytes']
    elif kind=='ME':
        reverse=np.full(out['shape'][0],-1,np.int64);reverse[selected]=np.arange(len(selected))
        for offset,pair in enumerate(g.maps['maps'][event['map_key']]['offsets']):
            active=reverse[pair[1]]>=0
            offsets.append(np.asarray(pair[0])[active]);targets.append(reverse[np.asarray(pair[1])[active]])
            base.append(weight_base+offset*weight['stride'][0]*weight['element_bytes'])
        wk=weight['stride'][-2]*weight['element_bytes'];wn=weight['stride'][-1]*weight['element_bytes']
    else:
        source_ids=(np.asarray(g.maps['indices'][index][0])[selected]
                    if kind=='ancestor_merge' or event.get('indexed_source',False) else selected)
        offsets=[source_ids];targets=[np.arange(len(selected))];base=[weight_base]
        if kind in ('linear','ancestor_merge'):wk=weight['stride'][-1]*weight['element_bytes'];wn=weight['stride'][-2]*weight['element_bytes']
        else:wk=weight['stride'][-2]*weight['element_bytes'];wn=weight['stride'][-1]*weight['element_bytes']
    pointer=np.cumsum([0]+[len(ids) for ids in offsets]).astype(np.int64)
    starts=g.address_starts(source,np.concatenate(offsets))
    return starts,np.concatenate(targets).astype(np.int64),pointer,np.asarray(base,np.int64),wk,wn


def execute(sim,index,selected,columns=None,column=0):
    """Execute bounded products; an optional strip keeps real weight strides."""
    if sim.c.get('direct_operand_stream',False):
        from .direct import execute as direct_execute,supported,translation_capacity
        if supported(sim.g.events[index]) and (not sim.c.get('direct_translated_tiles',False)
                or translation_capacity(sim.c,sim.g.events[index])):
            return direct_execute(sim,index,selected,columns,column)
        sim.stats['direct_fallback_operations']+=1
    g=sim.g;c=sim.c;memory=sim.memory;event=g.events[index];kind=event['kind']
    dense=dense_consumer(c,event)
    source,weight=event['inputs'][:2];out=event['outputs'][0]
    if event['kind']=='ancestor_merge':weight=event['inputs'][22]
    inner=source['shape'][-1];full_width=out['shape'][-1]
    width=full_width if columns is None else columns
    # Narrow neighborhoods cannot populate a complete physical N wave.
    # Keep their original scalar copy path; wide neighborhoods amortize the
    # shared paired controller over all arrays. No data/quality-derived gate.
    sparse_supply=(kind in ('spconv','ME') and c.get('consumer_copy_neighborhoods',False)
                   and width>=c['arrays']*c['array_cols'] and inner>=c['array_rows'])
    mapped_supply=(kind=='ancestor_merge' and c.get('consumer_feature_rhs_supply_active',False))
    supply=dense or sparse_supply or mapped_supply
    copy_batch=c.get('consumer_copy_batch_active',0)
    if c.get('consumer_copy_fit_weights',False) and (
            weight['role']!='weight' or weight['bytes']>c['weight_sram_bytes']):copy_batch=0
    if not dense and not sparse_supply:copy_batch=0
    weight_copy_batch=0 if c.get('consumer_copy_features_only',False) else copy_batch
    starts,targets,pointer,base,wk,wn=product_layout(g,index,selected)
    slots,capacity,output_budget=operand_capacity(c,out['element_bytes'])
    stream=c.get('stream_operand_slices',False) and not c.get('direct_operand_stream',False)
    if c.get('stream_on_calendar_pressure',False):
        slice_count=math.ceil(min(inner,c['k_tile'])*out['element_bytes']/SECTOR_BYTES)
        # Both slice reads plus full A/B fill and partial read/commit.
        stream=stream and c['arrays']*(2*slice_count+4)>c['queue_entries']
    interleave=stream and c.get('interleave_operand_slices',False)
    zero_seed=dense and c.get('consumer_zero_seed_active',False) and inner>0 and not stream
    packed=(dense and c.get('consumer_packed_operands_active',False) and
            c.get('reuse_operand_tiles',False) and not c.get('stream_operand_slices',False) and
            source['element_bytes']==weight['element_bytes']==out['element_bytes']==2)
    bk=c['k_tile']*(4//out['element_bytes'] if packed else 1)
    if dense:bk//=c.get('consumer_reduction_slices_active',1)
    # Only an admitted B operand batch receives this controller schedule.
    # The original public flags retain their old behavior for other callers.
    batch_prefetch=supply and c.get('coverage_operand_prefetch_active',False)
    warm_operands=c.get('prefetch_cached_operands',False) or batch_prefetch
    ready_order=c.get('operand_ready_order',False) or batch_prefetch
    if stream and (not c.get('reuse_operand_tiles',False) or not c.get('array_wave_pipeline',False)):
        raise ValueError('K streaming requires finite reusable operands and the existing array wave pipeline')
    if len(selected)*width*4>output_budget:raise ValueError('FP32 accumulator window exceeds output SRAM')
    # Parameters are normally in the weight SRAM; aten.mm can consume an
    # activation right-hand side, which belongs to the feature directory.
    which=int(weight['role']=='weight')
    # A generated RHS remains an actual feature, including its producer,
    # version, directory capacity and bank contention. The same registered
    # copy controller can alternate two sources in that one directory; its
    # combined victim guard must cover both admitted descriptor batches.
    feature_rhs=(supply and which==0 and c.get('consumer_feature_rhs_supply_active',False))
    banks,calendar,lengths=memory.ports(2)
    activity=None
    if sim.activity is not None:
        count=sum(math.ceil((pointer[i+1]-pointer[i])/c['array_rows']) for i in range(len(pointer)-1))
        count*=math.ceil(width/c['array_cols'])*math.ceil(inner/bk)
        activity=sim.activity.matrix_buffer(count*(math.ceil(min(inner,c['k_tile'])*out['element_bytes']/SECTOR_BYTES) if stream else 1))
    done,tiles,occupied,payload,reused,records,peak,batches,warmed,adjacent,preloaded=products(starts,targets,pointer,
        base+column*wn,wk,wn,source['stride'][-1]*source['element_bytes'],inner,width,len(selected),
        out['element_bytes'],c['fp32_mac_cycles'] if out['element_bytes']==4 else 1,
        c['array_rows'],c['array_cols'],bk,sim.feeder_ready,sim.matrix,c['array_wave_pipeline'],
        memory.directories[0],memory.directories[which],memory.slots[0],memory.slots[which],
        memory.links[0],memory.links[which],memory.meta[0],memory.meta[which],
        memory.state,memory.pending,banks,calendar,lengths,
        memory.priorities,c['victim_scan_entries'],memory.counts,
        c['cam_lookups_per_cycle'],c['cam_latency_cycles'],c['bank_bytes_per_cycle'],
        memory.burst_cycles,c['dram_latency_cycles'],c.get('independent_eviction_drain',False),
        c.get('reuse_operand_tiles',False),sim.feeder_b_ready,
        c.get('operand_k_outer',False) or (dense and c.get('dense_operand_k_outer',False)),memory.bank_layout,
        c.get('retain_sparse_accumulator',False) if kind in ('spconv','ME') else c.get('retain_dense_accumulator',False),
        c.get('separate_read_completion',False),
        memory.ports(0),memory.ports(which),sim.accumulator_base(out['element_bytes']),ready_order,activity,
        None if memory.directory_counts is None else memory.directory_counts[0],
        None if memory.directory_counts is None else memory.directory_counts[which],memory.dma_activity,
        stream,sim.input_latches,interleave,c.get('prefetch_microprogram_tiles',1),warm_operands,
        c.get('bounded_operand_copy',False),sim.copy_ready,memory.bank_layouts[which],
        supply and c.get('operand_group_k_outer',False),
        dense and c.get('dense_blocked_reduction',False),
        dense and c.get('consumer_weight_retention_active',False),packed,
        c['k_tile'] if dense and c.get('consumer_reduction_slices_active',1)>1 else 0,
        zero_seed,
        copy_batch,weight_copy_batch,c.get('consumer_copy_register_control',False),sim.copy_descriptor_queue,
        c.get('consumer_copy_ready_first',False),c.get('consumer_copy_source_credit',False),
        sim.copy_source_order,supply and c.get('consumer_operand_skew_active',False),
        supply and (which==1 or feature_rhs) and copy_batch>=2 and weight_copy_batch>=2 and not stream and
        c.get('consumer_dual_supply_active',False),
        supply and (which==1 or feature_rhs) and c.get('consumer_adjacent_supply_active',False),
        c.get('consumer_copy_mask_pipeline_active',False),feature_rhs)
    if activity is not None:
        if not stream and len(activity)!=tiles:raise AssertionError('Activity did not observe every product tile')
        sim.activity.add_matrix(activity[:records].copy(),done,tiles=int(tiles))
    if payload>capacity:raise AssertionError('Operand payload exceeds a physical feeder slot')
    sim.stats['feeder_matrix_tiles']+=int(tiles)
    sim.stats['feeder_peak_payload_bytes']=max(sim.stats['feeder_peak_payload_bytes'],int(payload))
    sim.stats['array_occupied_cycles']+=int(occupied)
    sim.stats['matrix_flops']+=g.flops(index,selected)*width//full_width
    if dense and kind!='linear':sim.stats['consumer_mm_batch_operations']+=1
    if feature_rhs:sim.stats['consumer_feature_rhs_supply_operations']+=1
    sim.stats['feeder_reserved_bytes']=slots*capacity
    sim.stats['feeder_reused_operand_bytes']+=int(reused)
    if c.get('consumer_adjacent_supply_active',False):
        sim.stats['consumer_adjacent_supply_batches']+=int(adjacent)
        sim.stats['consumer_adjacent_supply_preloaded_bytes']+=int(preloaded)
    if zero_seed:
        sim.stats['consumer_zero_seed_tiles']+=math.ceil(len(selected)/c['array_rows'])*math.ceil(width/c['array_cols'])
        sim.stats['consumer_zero_seed_control_cycles']+=math.ceil(len(selected)/c['array_rows'])*math.ceil(width/c['array_cols'])
    if warm_operands:sim.stats['operand_directory_prefetches']+=int(warmed)
    if c.get('bounded_operand_copy',False):sim.stats['operand_copy_register_bytes']=sim.copy_register_storage_bytes
    if c.get('consumer_copy_batch_active',0):sim.stats['consumer_copy_descriptor_bytes']=sim.copy_descriptor_storage_bytes
    if c.get('consumer_copy_batch_active',0) and c.get('consumer_copy_register_control',False):
        sim.stats['consumer_copy_register_descriptor_bytes']=sim.copy_descriptor_storage_bytes
    if c.get('stream_operand_slices',False):sim.stats['input_slice_latch_bytes']=input_latch_bytes(c)
    if c.get('interleave_operand_slices',False):
        sim.stats['input_microprogram_bytes']=stream_descriptor_bytes(c)
        sim.stats['input_microprogram_batches']+=int(batches)
        sim.stats['input_microprogram_peak_bank_entries']=max(sim.stats['input_microprogram_peak_bank_entries'],int(peak))
    return int(done)


def column_commit(sim,index,selected,column,width,finished):
    """Drain the bounded FP32 strip, add bias/cast, commit real output bytes."""
    g=sim.g;c=sim.c;m=sim.memory;event=g.events[index];out=event['outputs'][0]
    ready=sim.output_transfer(len(selected)*width*4,int(finished),offset=sim.accumulator_base(out['element_bytes']))
    passes=int(out['element_bytes']<4)
    if len(event['inputs'])>2+int(event.get('indexed_source',False)):
        bias=event['inputs'][2]
        if bias['bytes']:
            b=dict(bias,address=bias['address']+column*bias['stride'][-1]*bias['element_bytes'],
                   shape=[width],bytes=width*bias['element_bytes'])
            ready=max(ready,m.access(g.sectors(b,np.array([0],np.int64)),weight=bias['role']=='weight'))
            passes+=1
    cost=math.ceil(len(selected)*width/c['vector_lanes'])*passes
    ready=sim.issue_vector(ready,cost);sim.stats['vector_cycles']+=cost
    epilogue=sim.column_epilogues.get(index)
    if epilogue is not None:
        # Preserve the original materialized tensor's precision boundary.
        # Cast/bias precede a real narrow scratch write/read; GELU then uses
        # the existing vector unit, not an additional epilogue datapath.
        nbytes=len(selected)*width*out['element_bytes'];base=sim.accumulator_base(out['element_bytes'])
        ready=sim.output_transfer(nbytes,ready,write=True,offset=base)
        ready=sim.output_transfer(nbytes,ready,offset=base)
        from .protocol import GELU_PASSES
        cost=math.ceil(len(selected)*width/c['vector_lanes'])*GELU_PASSES
        ready=sim.issue_vector(ready,cost);sim.stats['vector_cycles']+=cost
        ready=sim.output_transfer(nbytes,ready,write=True,offset=base)
        ready=sim.output_transfer(nbytes,ready,offset=base)
        sim.stats['column_gelu_elements']+=len(selected)*width
        sim.stats['column_gelu_scratch_bytes']+=4*nbytes
        out=epilogue['outputs'][0]
    view=dict(out,address=out['address']+column*out['element_bytes'],
              shape=out['shape'][:-1]+[width],bytes=len(selected)*width*out['element_bytes'])
    done=sim.store(view,selected,ready)
    # Only this physical window waits for its last accumulator read/commit.
    # The other declared window may prefetch and compute in the meantime.
    if sim.c.get('column_window_overlap',False):sim.buffer_ready[sim.active_output_buffer]=done
    else:m.clock=max(m.clock,done)
    return int(done)


def indexed_window(sim,index,selected):
    """Read/decode real row IDs for this bounded M/N owner only.

    No decoded full-domain address table survives a window or N stripe.
    Repeated stripes probe ordinary cacheable index bytes again and pay the
    same vector address operation before requesting their source features.
    """
    event=sim.g.events[index]
    if not event.get('indexed_source',False):return
    memory=sim.memory
    ready=memory.access(sim.g.sectors(event['inputs'][-1],selected),ready=memory.clock)
    cost=math.ceil(len(selected)/sim.c['vector_lanes'])
    ready=sim.issue_vector(ready,cost)
    sim.stats['vector_cycles']+=cost
    sim.stats['indexed_affine_address_cycles']+=cost
    sim.stats['indexed_affine_window_reads']+=1
    memory.clock=max(memory.clock,ready)


def dense_columns(sim,index,selected):
    """Complete a bounded output-column strip before changing its owner."""
    event=sim.g.events[index];out=event['outputs'][0]
    if not dense_consumer(sim.c,event):raise ValueError('Dense strips require a rowwise dense product')
    n=out['shape'][-1];stripe=min(n,sim.c['dense_output_columns']);done=sim.cycle
    for column in range(0,n,stripe):
        width=min(stripe,n-column)
        if sim.c.get('column_window_overlap',False):
            sim.active_output_buffer=int(np.argmin(sim.buffer_ready))
            sim.memory.clock=max(sim.memory.clock,sim.cycle,int(sim.buffer_ready[sim.active_output_buffer]))
        indexed_window(sim,index,selected)
        finished=execute(sim,index,selected,columns=width,column=column)
        done=max(done,column_commit(sim,index,selected,column,width,finished))
        sim.stats['dense_column_strips']+=1
    sim.stats['dense_accumulator_peak_bytes']=max(sim.stats['dense_accumulator_peak_bytes'],len(selected)*stripe*4)
    return int(done)


def sparse_stripe(config,event):
    """Array-aligned N strip under the logical offset-weight byte budget.

    Strided layouts may touch more directory slots; real cache transactions
    still enforce capacity and charge their misses/evictions without packing.
    """
    n=event['outputs'][0]['shape'][-1];weight=event['inputs'][1]
    bytes_per_column=math.ceil(weight['bytes']/n)
    maximum=min(n,config['sparse_output_columns'],config['weight_sram_bytes']//bytes_per_column)
    if maximum<config['array_cols']:
        raise ValueError('One sparse output-channel tile cannot fit weight SRAM')
    return maximum//config['array_cols']*config['array_cols']


def use_sparse_columns(config,event,count):
    """Only oversized weights revisited across multiple M windows benefit."""
    if event['kind'] not in ('spconv','ME') or not config.get('sparse_output_columns',0):return False
    if event['inputs'][1]['bytes']<=config['weight_sram_bytes']:return False
    out=event['outputs'][0]
    _,_,budget=operand_capacity(config,out['element_bytes'])
    original_rows=budget//(out['shape'][-1]*4)//config['array_rows']*config['array_rows']
    return count>max(config['array_rows'],original_rows)


def dense_consumer(config,event):
    """Existing dense program; MM keeps its native RHS K/N strides and role."""
    if config.get('coverage_private_consumer_bypass_active',False):
        from .coverage_rows import CoverageRows
        if CoverageRows.private_rows(event):return False
    return event['kind']=='linear' or (config.get('consumer_mm_batches_active',False) and
        event.get('op')=='aten.mm.default')


def use_dense_outer(config,event):
    """Keep oversized dense weights in N stripes across finite row owners."""
    return (dense_consumer(config,event) and config.get('dense_outer_rows',False) and
            bool(config.get('dense_output_columns',0)) and
            event['inputs'][1]['bytes']>config['weight_sram_bytes'])


def sparse_columns(sim,index,selected):
    return column_rows(sim,index,selected,sparse=True)


def dense_outer_rows(sim,index,selected):
    return column_rows(sim,index,selected,sparse=False)


def consumer_resident_rows(config,events,limit):
    """Limit an admitted cohort by both input and output logical row bytes.

    This is an admission heuristic, never a cache-hit promise or pin. Strided
    rows and real old versions still use the original slot allocator/traffic.
    """
    if not config.get('consumer_resident_rows_active',False):return limit
    if not any(e['kind']=='linear' and e['inputs'][1]['role']=='weight' and
               e['inputs'][1]['bytes']>config['weight_sram_bytes'] for e in events):return limit
    versions={}
    for event in events:
        for spec in event['inputs']+event['outputs']:
            if spec['role']=='feature' and len(spec['shape'])==2:
                versions[(spec['storage'],spec['version'],spec['address'],tuple(spec['shape']),tuple(spec['stride']))]=spec
    per_row=sum(s['shape'][-1]*s['element_bytes'] for s in versions.values())
    if not per_row:return limit
    # Leave boundary slots for each logical version; actual sector/slot
    # fragmentation remains charged, especially under a noncontiguous order.
    capacity=min(config['feature_sram_bytes'],config['cam_entries']*SLOT_SECTORS*SECTOR_BYTES)
    available=capacity-len(versions)*SLOT_SECTORS*SECTOR_BYTES
    rows=available//per_row//config['array_rows']*config['array_rows']
    return min(limit,rows) if rows>=config['array_rows'] else limit


def column_rows(sim,index,selected,sparse):
    """Keep a weight stripe across bounded M windows and true offset/K order.

    N is outermost. Each row window still has its own FP32 partial sums and
    all offset/K contributions in their original order. No full output or
    unbudgeted operands are held; only committed features can survive a window.
    """
    g=sim.g;c=sim.c;m=sim.memory;event=g.events[index];out=event['outputs'][0]
    prefix='sparse' if sparse else 'dense_outer'
    if sparse and event['kind'] not in ('spconv','ME'):raise ValueError('Sparse strips require explicit offset edges')
    if not sparse and not dense_consumer(c,event):raise ValueError('Dense strips require a rowwise dense product')
    n=out['shape'][-1]
    stripe=sparse_stripe(c,event) if sparse else min(n,c['dense_output_columns'])
    _,_,budget=operand_capacity(c,out['element_bytes'])
    bm=min(c['queue_entries']*c['array_rows'],budget//(stripe*4))
    bm=bm//c['array_rows']*c['array_rows']
    if not sparse:bm=consumer_resident_rows(c,[event],bm)
    if bm<=0:raise ValueError('Column strip accumulator exceeds output SRAM')
    done=sim.cycle
    serpentine=(not sparse and c.get('dense_serpentine_columns',False) and
                event['inputs'][1]['role']=='weight' and
                event['inputs'][1]['bytes']>c['weight_sram_bytes'])
    columns=range(0,n,stripe)
    reverse=serpentine and getattr(sim,'chain_window_ordinal',0)%2
    if reverse:columns=reversed(columns)
    for column in columns:
        if serpentine:
            # Registered direction/column address before any stripe reads.
            # The actual finite cache decides whether a tail survived.
            m.clock=max(m.clock,sim.cycle)+c['issue_cycles']
            sim.stats['dense_serpentine_control_cycles']+=c['issue_cycles']
            sim.stats['dense_serpentine_reverse_strips']+=int(bool(reverse))
        width=min(stripe,n-column)
        for ids in sim.groups(index,selected,bm):
            buffer=int(np.argmin(sim.buffer_ready))
            sim.active_output_buffer=buffer
            m.clock=max(m.clock,sim.cycle,int(sim.buffer_ready[buffer]))
            # Captured edges are topology, not free rulebook data. Each new
            # window/stripe reads its actual row indices before feature issue.
            if sparse:
                rule=g.maps['maps'][event['map_key']];base=g.map_bases[event['map_key']]
            if sparse and event['kind']=='spconv':
                mapping=np.asarray(rule['mapping'])
                address=(np.arange(mapping.shape[0])[:,None]*mapping.shape[1]+ids)*mapping.itemsize+base
                sectors=np.unique(address//SECTOR_BYTES)
            elif sparse:
                active=np.zeros(out['shape'][0],bool);active[ids]=True;addresses=[]
                for offset,pair in enumerate(rule['offsets']):
                    edges=np.flatnonzero(active[pair[1]])
                    for axis,values in enumerate(pair):
                        start=base+int(rule['pair_byte_offsets'][offset])+(pair[0].nbytes if axis else 0)
                        addresses.append((start+edges*values.itemsize)//SECTOR_BYTES)
                sectors=np.unique(np.concatenate(addresses))
            if sparse:m.clock=max(m.clock,m.access(sectors))
            indexed_window(sim,index,ids)
            finished=execute(sim,index,ids,columns=width,column=column)
            committed=column_commit(sim,index,ids,column,width,finished)
            done=max(done,committed)
            sim.buffer_ready[buffer]=committed if c.get('column_window_overlap',False) else done
            sim.stats[prefix+'_column_windows']+=1
            sim.stats[prefix+'_accumulator_peak_bytes']=max(sim.stats[prefix+'_accumulator_peak_bytes'],len(ids)*width*4)
            m.clock+=c['issue_cycles']
        sim.stats[prefix+'_column_strips']+=1
    return int(done)
