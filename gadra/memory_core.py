"""Compiled transition kernel for the Python SRAM/CAM/DMA state machine.

The same Python source runs without Numba. Compilation only removes interpreter
overhead; allocation, LRU, finite outstanding requests and bank timing are equal.
"""
import numpy as np
from .protocol import SECTOR_BYTES,SLOT_SECTORS,DMA_BURST_BYTES,BURST_SECTORS,DRAM_TIME_TICKS
try:
    from numba import njit,types
    from numba.typed import Dict,List
    COMPILED=True
except ImportError:
    COMPILED=False
    def njit(*args,**kwargs):
        return lambda function:function


def new_directory():
    return Dict.empty(key_type=types.int64,value_type=types.int64) if COMPILED else {}


def new_intervals():
    """Host observation storage, never a DMA admission or readiness input."""
    return List.empty_list(types.UniTuple(types.int64,2)) if COMPILED else []


@njit(cache=True)
def matrix_tiles(availability,count,duration,earliest,initiation=0,activity=None):
    """Issue every padded tile to the first available physical array."""
    done=earliest
    for tile in range(count):
        unit=np.argmin(availability)
        begin=max(earliest,availability[unit])
        done=max(done,begin+duration)
        availability[unit]=begin+(initiation if initiation else duration)
        if activity is not None:
            activity[tile,0]=begin
            activity[tile,1]=availability[unit]
            activity[tile,2]=unit
    return done


@njit(cache=True)
def transaction(earliest,write,state,pending,stats,burst_cycles,latency,sector_bytes,activity=None):
    which=np.argmin(pending)
    issued=max(earliest,pending[which])
    # DDR and core are different clock domains. Preserve fractional core
    # cycles across bursts rather than rounding each burst to a whole cycle.
    begin=max(issued*DRAM_TIME_TICKS,state[1])
    state[1]=begin+burst_cycles
    if activity is not None:
        # The single DDR bus issues chronologically. Coalesce contiguous
        # service without dropping fractional cycles or intervening gaps.
        if len(activity) and activity[-1][1]==begin:
            activity[-1]=(activity[-1][0],state[1])
        else:activity.append((begin,state[1]))
    finish=(state[1]+DRAM_TIME_TICKS-1)//DRAM_TIME_TICKS+latency
    pending[which]=finish
    stats[1 if write else 0]+=sector_bytes
    stats[2]+=1
    return finish


@njit(cache=True)
def reserve_bank(bank,earliest,duration,clock,calendar,lengths):
    n=lengths[bank]
    remove=0
    while remove<n and calendar[bank,remove,1]<=clock:remove+=1
    if remove:
        for j in range(n-remove):calendar[bank,j]=calendar[bank,j+remove]
        n-=remove
    begin=earliest
    pos=0
    while pos<n:
        if begin+duration<=calendar[bank,pos,0]:break
        if begin<calendar[bank,pos,1]:begin=calendar[bank,pos,1]
        pos+=1
    if n>=calendar.shape[1]:
        # Backpressure on a full calendar: append after all reservations.
        # Inserting into a former hole after retiring an entry could overlap
        # the next reservation when that retirement moves the start forward.
        begin=max(begin,calendar[bank,n-1,1]);pos=n
        for j in range(1,n):calendar[bank,j-1]=calendar[bank,j]
        n-=1;pos=max(0,pos-1)
    for j in range(n,pos,-1):calendar[bank,j]=calendar[bank,j-1]
    calendar[bank,pos,0]=begin;calendar[bank,pos,1]=begin+duration
    lengths[bank]=n+1
    return begin+duration


@njit(cache=True)
def reserve_copy_bank(bank,earliest,duration,state,calendar,lengths,destination=False):
    """Admit a copy beat only when its finite bank queue has a free entry.

    The producer stalls to the first retiring request, not the final booked
    request. Already issued array/DMA work keeps running across this wait.
    Advancing the shared dispatch frontier forbids later requests from filling
    holes in retired history. No live reservation is dropped on admission.
    """
    if lengths[bank]==calendar.shape[1]:
        if len(state)>2:
            # Host-only attribution: never consulted by admission or timing.
            delay=max(0,calendar[bank,0,1]-state[0])
            field=4 if destination else 2
            state[field]+=int(delay>0)
            state[field+1]+=delay
        state[0]=max(state[0],calendar[bank,0,1])
    return reserve_bank(bank,max(earliest,state[0]),duration,state[0],calendar,lengths)


@njit(cache=True)
def masked_transfer(slot,mask,earliest,state,banks,calendar,lengths,bank_width,striped):
    """Transfer byte-enabled sectors under the selected physical bank layout.

    Directory allocation remains 2 KiB. Layout 0 is slot-banked, 1 preserves
    legacy physical-sector striping, and 2 skews sectors by the slot number.
    All layouts use the same finite banks and transfer every enabled sector.
    """
    demand=np.zeros(len(banks),np.int64)
    count=0
    for bit in range(SLOT_SECTORS):
        if mask&(np.uint64(1)<<np.uint64(bit)):
            address=slot*SLOT_SECTORS+bit
            # Skew the bank column by its physical row. For B=64 this is
            # (slot+bit)%B; the row/column permutation is bijective for any B.
            bank=((address//len(banks)+address) if striped==2 else address if striped else slot)%len(banks)
            demand[bank]+=SECTOR_BYTES;count+=1
    done=earliest
    for bank in range(len(banks)):
        if demand[bank]:
            end=reserve_bank(bank,earliest,(demand[bank]+bank_width-1)//bank_width,state[0],calendar,lengths)
            banks[bank]=max(banks[bank],end);done=max(done,end)
    return done,count


@njit(cache=True)
def copy_transfer(slot,tag,mask,earliest,state,banks,calendar,lengths,bank_width,striped,sink,
                  defer_source=None):
    """One sector register per physical source bank, held through sink writes.

    No source read can overwrite a busy route register. Every byte-enabled
    destination sector uses the declared output banks, including transpose
    scatter and fanout. Cache ownership survives the actual source read.
    """
    keys,pointer,targets,lane_ready,out_banks,out_calendar,out_lengths,target_skew=sink
    source_done=earliest;done=earliest;read_bytes=written_bytes=0
    for bit in range(SLOT_SECTORS):
        if not mask&(np.uint64(1)<<np.uint64(bit)):continue
        key=tag*SLOT_SECTORS+bit
        where=np.searchsorted(keys,key)
        if where>=len(keys) or keys[where]!=key:raise AssertionError('Missing actual copy sector')
        address=slot*SLOT_SECTORS+bit
        bank=((address//len(banks)+address) if striped==2 else address if striped else slot)%len(banks)
        group=np.argmin(lane_ready[bank])
        begin=max(state[0],earliest,lane_ready[bank,group])
        if defer_source is not None:
            # Only admitted descriptors can retain an unissued source. A
            # full calendar appends after its tail and stays full, preventing
            # backfill into forgotten history without stalling other banks.
            begin=max(begin,banks[bank],defer_source[bank])
            returned=reserve_bank(bank,begin,(SECTOR_BYTES+bank_width-1)//bank_width,
                                  state[0],calendar,lengths)
            defer_source[bank]=returned
        else:
            returned=reserve_copy_bank(bank,begin,(SECTOR_BYTES+bank_width-1)//bank_width,
                                       state,calendar,lengths)
        banks[bank]=max(banks[bank],returned);source_done=max(source_done,returned)
        read_bytes+=SECTOR_BYTES;released=returned+1 # registered bank-to-bank route
        for edge in range(pointer[where],pointer[where+1]):
            target=targets[edge]
            out_bank=((target//len(out_banks)+target) if target_skew else target)%len(out_banks)
            written=reserve_copy_bank(out_bank,returned+1,(SECTOR_BYTES+bank_width-1)//bank_width,
                                      state,out_calendar,out_lengths,True)
            out_banks[out_bank]=max(out_banks[out_bank],written)
            released=max(released,written);written_bytes+=SECTOR_BYTES
        lane_ready[bank,group]=released;done=max(done,released)
    return source_done,done,read_bytes,written_bytes


@njit(cache=True)
def access_base(tags,masks,directory,slots,links,meta,state,pending,banks,calendar,bank_lengths,priorities,victim_scan,stats,
           write,ready,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,sector_bytes,
           independent_eviction_drain=False,striped=False,separate_read_completion=False,profile=None,activity=None,prefetch=False,
           sink=None):
    if prefetch and write:raise ValueError('A directory prefetch cannot produce data')
    if sink is not None and (prefetch or write):raise ValueError('A copy route requires actual source reads')
    before=stats.copy() if profile is not None else None
    completion=state[0];routed_writes=0
    state[0]+=(len(tags)+lookup_width-1)//lookup_width+cam_latency
    capacity=len(slots)
    masks_view=slots.view(np.uint64)
    for pos in range(len(tags)):
        tag=tags[pos];mask=masks[pos]
        if tag in directory:
            slot=directory[tag];stats[3]+=1
            # Remove from LRU list before appending as newest.
            previous=links[slot,0];following=links[slot,1]
            if previous>=0:links[previous,1]=following
            else:meta[1]=following
            if following>=0:links[following,0]=previous
            else:meta[2]=previous
        else:
            stats[4]+=1
            replacement_ready=state[0]
            if meta[0]<capacity:
                slot=meta[0];meta[0]+=1
            else:
                slot=meta[1];stats[5]+=1
                candidate=slot
                for _ in range(min(victim_scan,capacity)):
                    if slots[candidate,4]==0:slot=candidate;break
                    candidate=links[candidate,1]
                    if candidate<0:break
                oldtag=slots[slot,0]
                del directory[oldtag]
                previous=links[slot,0];following=links[slot,1]
                if previous>=0:links[previous,1]=following
                else:meta[1]=following
                if following>=0:links[following,0]=previous
                else:meta[2]=previous
                replacement_ready=max(state[0],slots[slot,3])
                if not independent_eviction_drain:state[0]=replacement_ready
                for burst in range(SLOT_SECTORS//BURST_SECTORS):
                    bits=np.uint64((1<<BURST_SECTORS)-1)<<np.uint64(burst*BURST_SECTORS)
                    if masks_view[slot,2]&bits:
                        dirty_bits=masks_view[slot,2]&bits
                        end,count=masked_transfer(slot,dirty_bits,max(state[0],slots[slot,3]),
                            state,banks,calendar,bank_lengths,bank_width,striped)
                        if not independent_eviction_drain:state[0]=max(state[0],end)
                        stats[6]+=count*sector_bytes
                        returned=transaction(end,True,state,pending,stats,burst_cycles,dram_latency,DMA_BURST_BYTES,activity)
                        replacement_ready=max(replacement_ready,returned)
            slots[slot,0]=tag;slots[slot,1]=0;slots[slot,2]=0
            slots[slot,3]=replacement_ready if independent_eviction_drain else 0
            slots[slot,5]=slots[slot,3]
            directory[tag]=slot
        links[slot,0]=meta[2];links[slot,1]=-1
        if meta[2]>=0:links[meta[2],1]=slot
        else:meta[1]=slot
        meta[2]=slot
        valid=masks_view[slot,1];dirty=masks_view[slot,2]
        # Data readiness and slot ownership are distinct. Independent readers
        # may overlap, but eviction still waits for every reserved bank read.
        # Writes retain conservative write-after-read ordering for the slot.
        available=slots[slot,5] if separate_read_completion and not write else slots[slot,3]
        if write:
            valid|=mask;dirty|=mask;available=max(available,ready)
        else:
            missing=mask&~valid
            for burst in range(SLOT_SECTORS//BURST_SECTORS):
                bits=np.uint64((1<<BURST_SECTORS)-1)<<np.uint64(burst*BURST_SECTORS)
                if missing&bits:
                    arrived=transaction(max(state[0],slots[slot,3]) if independent_eviction_drain else state[0],False,state,pending,stats,burst_cycles,dram_latency,DMA_BURST_BYTES,activity)
                    # Masked fill preserves dirty bytes already produced in
                    # this burst; it also consumes the actual SRAM write port.
                    fill=bits&~dirty
                    end,count=masked_transfer(slot,fill,arrived,state,banks,calendar,bank_lengths,bank_width,striped)
                    available=max(available,end)
                    stats[7]+=count*sector_bytes
                    valid|=bits
        # Reserve each actual bank transfer before another request can evict
        # its owner. Independent banks may start as their operands arrive.
        # Consumer-ready can delay the payload read while a cold fill starts
        # now. The reserved bank read extends slot ownership until it ends.
        # A cold prefetch populates the existing directory only. Its real
        # fill ports and owner remain live, but it neither reads a consumer
        # payload nor claims an operand half. Consumption must probe again.
        if prefetch:bank_end=max(state[0],available);count=0;consumed=bank_end
        elif sink is not None:
            bank_end,consumed,nread,nwrite=copy_transfer(slot,tag,mask,max(state[0],available,ready),
                state,banks,calendar,bank_lengths,bank_width,striped,sink)
            count=nread//sector_bytes;stats[7]+=nwrite;routed_writes+=nwrite
        else:
            bank_end,count=masked_transfer(slot,mask,max(state[0],available,ready),state,banks,calendar,bank_lengths,bank_width,striped)
            consumed=bank_end
        completion=max(completion,consumed)
        slots[slot,5]=max(available,bank_end) if write else available
        masks_view[slot,1]=valid;masks_view[slot,2]=dirty
        slots[slot,3]=max(slots[slot,3],available,bank_end)
        slots[slot,4]=priorities.get(tag,0)
        stats[7 if write else 6]+=count*sector_bytes
    if profile is not None:
        profile+=stats-before
        profile[7]-=routed_writes # destination is the output store, not this directory
    return completion


@njit(cache=True)
def copy_ready_sector(owners,tags,masks,bit,ready,state,slots,banks,calendar,lengths,
                      bank_width,striped,sink,lookup_width,source_returns,stats,profile):
    """Drain one bounded sector wave using producer/credit ready signals.

    Completion times implement event notification, not future-priority keys.
    Both passes pay a descriptor scan and a registered ready-mask selection.
    No other directory client can allocate while these references are live.
    """
    keys,pointer,targets,lane_ready,out_banks,out_calendar,out_lengths,target_skew=sink
    pending=np.zeros(len(owners),np.bool_)
    flag=np.uint64(1)<<np.uint64(bit)
    for at in range(len(owners)):pending[at]=bool(masks[at]&flag)
    done=state[0]
    for phase in range(2):
        if not np.any(pending):break
        state[0]+=(len(owners)+lookup_width-1)//lookup_width+1
        for at in range(len(owners)):
            if not pending[at]:continue
            slot=owners[at];address=slot*SLOT_SECTORS+bit
            bank=((address//len(banks)+address) if striped==2 else address if striped else slot)%len(banks)
            available=max(ready,slots[slot,5],np.min(lane_ready[bank]))
            if lengths[bank]==calendar.shape[1]:available=max(available,calendar[bank,0,1])
            if phase==0 and available>state[0]:continue
            # The second pass uses the original finite reservation path.
            # Data waits in its source owner; do not drain all producers
            # before admitting the next sector's independent ready requests.
            source,end,nread,nwrite=copy_transfer(slot,tags[at],flag,max(state[0],ready,slots[slot,5]),
                state,banks,calendar,lengths,bank_width,striped,sink)
            slots[slot,3]=max(slots[slot,3],source)
            source_returns[at]=max(source_returns[at],source)
            stats[6]+=nread;stats[7]+=nwrite
            if profile is not None:profile[6]+=nread
            pending[at]=False;done=max(done,end)
    return done


@njit(cache=True)
def copy_credit_batch(owners,tags,masks,ready,state,slots,banks,calendar,lengths,
                      bank_width,striped,sink,lookup_width,source_returns,stats,profile):
    """Retry source sectors in the existing, finite descriptor remaining masks.

    A request remains in its descriptor until its data, source-bank credit,
    and return lane are ready. Only then does copy_transfer issue the actual
    read. Target fanout retains the original blocking admission semantics.
    Event jumps occur only after scanning all admitted descriptors; they are
    wakeups, never future semantic priorities or extra request queues.
    """
    remaining=masks.copy()  # Existing 64-bit descriptor mask, not extra RAM.
    keys,pointer,targets,lanes,out_banks,out_calendar,out_lengths,target_skew=sink
    done=state[0]
    while np.any(remaining):
        # Registered next-set-bit, credit/lane test, then bounded issue scan.
        state[0]+=2+(len(owners)+lookup_width-1)//lookup_width
        issued=False;wake=np.iinfo(np.int64).max
        for at in range(len(owners)):
            if not remaining[at]:continue
            bit=0
            while not remaining[at]&(np.uint64(1)<<np.uint64(bit)):bit+=1
            slot=owners[at];address=slot*SLOT_SECTORS+bit
            bank=((address//len(banks)+address) if striped==2 else address if striped else slot)%len(banks)
            available=max(ready,slots[slot,5],np.min(lanes[bank]))
            if lengths[bank]==calendar.shape[1]:available=max(available,calendar[bank,0,1])
            if available>state[0]:
                wake=min(wake,available)
                continue
            flag=np.uint64(1)<<np.uint64(bit)
            source,end,nread,nwrite=copy_transfer(slot,tags[at],flag,state[0],
                state,banks,calendar,lengths,bank_width,striped,sink)
            slots[slot,3]=max(slots[slot,3],source)
            source_returns[at]=max(source_returns[at],source)
            stats[6]+=nread;stats[7]+=nwrite
            if profile is not None:profile[6]+=nread
            remaining[at]&=~flag;issued=True;done=max(done,end)
        if not issued:
            if wake==np.iinfo(np.int64).max:raise AssertionError('Pending copy without wakeup')
            state[0]=max(state[0],wake)
    return done


@njit(cache=True)
def access(tags,masks,directory,slots,links,meta,state,pending,banks,calendar,bank_lengths,priorities,victim_scan,stats,
           write,ready,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,sector_bytes,
           independent_eviction_drain=False,striped=False,separate_read_completion=False,profile=None,activity=None,prefetch=False,
           sink=None,copy_batch=0,copy_control_registers=False,copy_queue=None,copy_ready_first=False,
           copy_source_credit=False,copy_source_defer=None,copy_mask_pipeline=False):
    """Optionally interleave actual source reads in a bounded admitted batch.

    The batch owns only directory slot references, never returned payloads.
    Existing bank route registers retain every returned byte until its write.
    """
    if not copy_batch or sink is None or write or prefetch:
        if sink is not None and copy_source_defer is not None:state[0]=max(state[0],np.max(copy_source_defer))
        result=access_base(tags,masks,directory,slots,links,meta,state,pending,banks,calendar,bank_lengths,
            priorities,victim_scan,stats,write,ready,lookup_width,cam_latency,bank_width,burst_cycles,
            dram_latency,sector_bytes,independent_eviction_drain,striped,separate_read_completion,profile,activity,prefetch,sink)
        if sink is not None and copy_source_defer is not None:
            for bank in range(len(copy_source_defer)):copy_source_defer[bank]=max(copy_source_defer[bank],result)
        return result
    # With no other directory client, newest batch members stay beyond the
    # old victim scan. Otherwise use the original individually owned path.
    batch=min(copy_batch,len(slots)-victim_scan-1)
    if copy_queue is not None:batch=min(batch,len(copy_queue)-1)
    if batch<2:
        if copy_source_defer is not None:state[0]=max(state[0],np.max(copy_source_defer))
        result=access_base(tags,masks,directory,slots,links,meta,state,pending,banks,calendar,bank_lengths,
            priorities,victim_scan,stats,write,ready,lookup_width,cam_latency,bank_width,burst_cycles,
            dram_latency,sector_bytes,independent_eviction_drain,striped,separate_read_completion,profile,activity,prefetch,sink)
        if copy_source_defer is not None:
            for bank in range(len(copy_source_defer)):copy_source_defer[bank]=max(copy_source_defer[bank],result)
        return result
    done=state[0];view=slots.view(np.uint64)
    for begin in range(0,len(tags),batch):
        stop=min(begin+batch,len(tags));size=stop-begin
        # Fill/admit through the original CAM, finite DMA and bank ports.
        # No payload read is performed before the bounded descriptors exist.
        access_base(tags[begin:stop],masks[begin:stop],directory,slots,links,meta,state,pending,
            banks,calendar,bank_lengths,priorities,victim_scan,stats,False,0,lookup_width,
            cam_latency,bank_width,burst_cycles,dram_latency,sector_bytes,independent_eviction_drain,
            striped,separate_read_completion,profile,activity,True,None)
        descriptors=np.empty(size,np.int64)
        if copy_queue is not None:
            cursor=copy_queue[-1];capacity=len(copy_queue)-1
            for at in range(size):
                entry=(cursor+at)%capacity;descriptors[at]=entry
                state[0]=max(state[0],copy_queue[entry])
            # Reclaim only after every selected descriptor's last source
            # response. Until reclamation its old fields remain intact.
            state[0]+=(size+lookup_width-1)//lookup_width
        # Recheck ownership after all allocations and write 32B descriptors.
        state[0]+=(size+lookup_width-1)//lookup_width+cam_latency
        state[0]+=(size*8+lookup_width-1)//lookup_width
        owners=np.empty(size,np.int64)
        for at in range(size):
            tag=tags[begin+at]
            if tag not in directory:raise AssertionError('Admitted copy owner was evicted')
            slot=directory[tag];owners[at]=slot
            if slots[slot,0]!=tag or masks[begin+at]&~view[slot,1]:
                raise AssertionError('Admitted copy owner has incomplete fill')
        stats[3]+=size
        if profile is not None:profile[3]+=size
        last_read=state[0]
        source_returns=np.full(size,state[0],np.int64)
        # The same paid descriptors can be mapped to a finite register queue.
        # Its mask OR tree is pipelined one binary level per cycle; the
        # sector counter then reads all descriptor fields without SRAM ports.
        enabled=np.uint64(0)
        if copy_control_registers:
            for at in range(size):enabled|=masks[begin+at]
            levels=0;leaves=1
            while leaves<size:leaves*=2;levels+=1
            state[0]+=levels
            # Select/update the remaining mask in stage 1; the existing
            # descriptor fanout is stage 2. One selected-sector register
            # permits the next selection to overlap the current fanout.
            if copy_mask_pipeline and enabled:state[0]+=1
        if copy_source_credit:
            end=copy_credit_batch(owners,tags[begin:stop],masks[begin:stop],ready,state,
                slots,banks,calendar,bank_lengths,bank_width,striped,sink,lookup_width,
                source_returns,stats,profile)
            last_read=max(last_read,np.max(source_returns));done=max(done,end)
        for bit in range(0 if copy_source_credit else SLOT_SECTORS):
            # Read the current bounded descriptors through control ports.
            # Cursor is one batch counter; inactive lanes never read payload.
            if copy_control_registers:
                if not enabled&(np.uint64(1)<<np.uint64(bit)):continue
                # Next-set-bit selection is part of this first registered
                # control stage, not a zero-cycle walk across empty sectors.
                state[0]+=(0 if copy_mask_pipeline else 1)+(size+lookup_width-1)//lookup_width
                if copy_source_defer is not None:state[0]+=1 # registered credit retry/selection
            else:state[0]+=(size*8+lookup_width-1)//lookup_width
            if copy_ready_first:
                end=copy_ready_sector(owners,tags[begin:stop],masks[begin:stop],bit,ready,state,
                    slots,banks,calendar,bank_lengths,bank_width,striped,sink,lookup_width,
                    source_returns,stats,profile)
                last_read=max(last_read,np.max(source_returns));done=max(done,end)
                continue
            for at in range(size):
                mask=masks[begin+at]&(np.uint64(1)<<np.uint64(bit))
                if not mask:continue
                slot=owners[at];tag=tags[begin+at]
                prior=source_returns[at] if copy_source_defer is not None else 0
                source,end,nread,nwrite=copy_transfer(slot,tag,mask,max(state[0],ready,slots[slot,5],prior),
                    state,banks,calendar,bank_lengths,bank_width,striped,sink,copy_source_defer)
                slots[slot,3]=max(slots[slot,3],source)
                last_read=max(last_read,source)
                source_returns[at]=max(source_returns[at],source)
                stats[6]+=nread;stats[7]+=nwrite
                if profile is not None:profile[6]+=nread
                done=max(done,end)
        # Invalidate/release the descriptor words; route payload registers
        # remain owned until their actual sink writes complete.
        if copy_queue is None:
            state[0]=max(state[0],last_read)+(size+lookup_width-1)//lookup_width
        else:
            for at in range(size):copy_queue[descriptors[at]]=source_returns[at]
            copy_queue[-1]=(cursor+size)%capacity
            state[0]+=1 # registered circular allocation cursor update
    return done


@njit(cache=True)
def flush(slots,meta,state,pending,stats,burst_cycles,latency,sector_bytes,banks,calendar,lengths,bank_width,striped=False,profile=None,activity=None):
    before=stats.copy() if profile is not None else None
    ready=max(state[0],np.max(pending))
    masks_view=slots.view(np.uint64)
    for slot in range(meta[0]):
        for burst in range(SLOT_SECTORS//BURST_SECTORS):
            bits=np.uint64((1<<BURST_SECTORS)-1)<<np.uint64(burst*BURST_SECTORS)
            if masks_view[slot,2]&bits:
                end,count=masked_transfer(slot,masks_view[slot,2]&bits,
                    max(state[0],slots[slot,3],np.min(pending)),state,banks,calendar,lengths,bank_width,striped)
                stats[6]+=count*sector_bytes
                ready=max(ready,transaction(end,True,state,pending,stats,burst_cycles,latency,DMA_BURST_BYTES,activity))
    if profile is not None:profile+=stats-before
    return ready


@njit(cache=True)
def discard(tags,masks,directory,slots,hold_until):
    saved=0
    masks_view=slots.view(np.uint64)
    for i in range(len(tags)):
        tag=tags[i]
        if tag not in directory:continue
        slot=directory[tag]
        dead=masks_view[slot,2]&masks[i]
        for bit in range(SLOT_SECTORS):
            if dead&(np.uint64(1)<<np.uint64(bit)):saved+=SECTOR_BYTES
        masks_view[slot,2]&=~masks[i]
        slots[slot,3]=max(slots[slot,3],hold_until)
    return saved


@njit(cache=True)
def resident(sectors,directory,slots):
    hits=0
    masks_view=slots.view(np.uint64)
    for sector in sectors:
        tag=sector//SLOT_SECTORS
        if tag in directory:
            slot=directory[tag]
            if masks_view[slot,1]&(np.uint64(1)<<np.uint64(sector%SLOT_SECTORS)):hits+=1
    return hits


@njit(cache=True)
def resident_tag_masks(tags,directory,slots):
    """One directory probe per tag; the ready wavefront consumes its mask."""
    masks=np.zeros(len(tags),np.uint64)
    view=slots.view(np.uint64)
    for i in range(len(tags)):
        if tags[i] in directory:masks[i]=view[directory[tags[i]],1]
    return masks


@njit(cache=True)
def masked_hits(requested,resident):
    """64-bit population count via 17 scalar shift/logic/add operations."""
    total=0
    for i in range(len(requested)):
        value=requested[i]&resident[i]
        value=value-((value>>np.uint64(1))&np.uint64(0x5555555555555555))
        value=(value&np.uint64(0x3333333333333333))+((value>>np.uint64(2))&np.uint64(0x3333333333333333))
        value=(value+(value>>np.uint64(4)))&np.uint64(0x0f0f0f0f0f0f0f0f)
        value=value+(value>>np.uint64(8))
        value=value+(value>>np.uint64(16))
        value=value+(value>>np.uint64(32))
        total+=int(value&np.uint64(0x7f))
    return total


@njit(cache=True)
def refresh_priority(slots,meta,priorities):
    for i in range(meta[0]):slots[i,4]=priorities.get(slots[i,0],0)
