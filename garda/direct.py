"""Finite bank-to-array streaming, without an intermediate operand-SRAM copy.

Each array owns one output tile and two sector-wide input latch groups. The
next group loads while the previous group feeds MACs. Directory fills, bank
conflicts, lookup admission, route latency and partial-sum commits remain real.
"""
import math

import numpy as np

from . import memory_core as core
from .feeder import accumulator,input_latch_bytes,operand_capacity,product_layout,rectangle
from .protocol import SECTOR_BYTES,DMA_BURST_BYTES,SLOT_BYTES,SLOT_SECTORS


(ROW,END,COLUMN,WIDTH,CURSOR,READY,ISSUED,VALID)=range(8)
DESCRIPTOR_BYTES=8*np.dtype(np.int64).itemsize


def descriptor_bytes(config):
    return (config['arrays']*DESCRIPTOR_BYTES+(16 if config.get('direct_multicast',False) else 0)
            if config.get('direct_operand_stream',False) else 0)


def translation_bytes(config):
    """Bounded tag/slot/mask records for one K segment per physical array."""
    if not config.get('direct_translated_tiles',False):return 0
    spans=math.ceil(config['k_tile']*4/SLOT_BYTES)+1
    return config['arrays']*(config['array_rows']+config['array_cols'])*spans*24


def translation_capacity(config,event):
    """Leave an entire victim scan outside the closed segment's working set.

    Otherwise use the original bounded feeder BEFORE issuing any direct work.
    Newest segment tags cannot be scanned while this many older slots remain.
    There are no other directory clients during the synchronous segment loop.
    """
    spans=math.ceil(min(config['k_tile'],event['inputs'][0]['shape'][-1])*
                    event['outputs'][0]['element_bytes']/SLOT_BYTES)+1
    a=config['arrays']*config['array_rows']*spans
    b=config['arrays']*config['array_cols']*spans
    weight=event['inputs'][22] if event['kind']=='ancestor_merge' else event['inputs'][1]
    feature=min(config['cam_entries'],config['feature_sram_bytes']//SLOT_BYTES)
    other=config['weight_sram_bytes']//SLOT_BYTES if weight['role']=='weight' else feature
    return (a+(b if weight['role']!='weight' else 0)+config['victim_scan_entries']<feature
            and b+config['victim_scan_entries']<other)


def supported(event):
    """A sector per lane suffices only for contiguous K operands.

    A transposed/gathered K slice needs additional registered compaction;
    keep its existing bounded feeder rather than invent that datapath.
    """
    source,weight=event['inputs'][:2];out=event['outputs'][0]
    if event['kind']=='ancestor_merge':weight=event['inputs'][22]
    k_axis=-1 if event['kind'] in ('linear','ancestor_merge','spconv') else -2
    return (source['stride'][-1]==1 and weight['stride'][k_axis]==1
            and source['element_bytes']==weight['element_bytes']==out['element_bytes'])


@core.njit(cache=True)
def row_entries(starts,depth,stride,element):
    """One paid translation field for every actual row/slot intersection."""
    total=0
    for start in starts:
        total+=(start+(depth-1)*stride+element-1)//SLOT_BYTES-start//SLOT_BYTES+1
    return total


@core.njit(cache=True)
def products(starts,targets,pointer,weight_bases,wk,wn,source_stride,inner,columns,
             output_rows,element_bytes,precision,bm,bn,arrays,latches,
             feature_directory,weight_directory,feature_slots,weight_slots,
             feature_links,weight_links,feature_meta,weight_meta,state,pending,
             banks,calendar,lengths,feature_ports,weight_ports,priorities,victim_scan,
             counts,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
             independent_drain,striped,weight_striped,separate_read_completion,
             accumulator_base,feature_profile,weight_profile,dma_activity,activity,
             issue_cycles,multicast,prefetch_bytes):
    fb,fc,fl=feature_ports;wb,wc,wl=weight_ports
    quantum=SECTOR_BYTES//element_bytes
    partial=np.zeros((output_rows,(columns+bn-1)//bn),np.int64)
    initial=accumulator(np.arange(output_rows),0,columns,columns,state[0],
        state,banks,calendar,lengths,bank_width,counts,True,accumulator_base)
    partial[:,:]=initial
    programs=np.zeros((len(arrays),8),np.int64)
    completion=initial;tiles=0;occupied=0;records=0;admissions=0;peak=0;prefetches=0;saved=0
    ntiles=(columns+bn-1)//bn
    for offset in range(len(pointer)-1):
        total=((pointer[offset+1]-pointer[offset]+bm-1)//bm)*ntiles
        admitted=0
        while True:
            for unit in range(len(arrays)):
                if programs[unit,VALID] or admitted>=total:continue
                row=pointer[offset]+(admitted//ntiles)*bm
                end=min(row+bm,pointer[offset+1]);column=(admitted%ntiles)*bn
                width=min(bn,columns-column);target=targets[row:end]
                state[0]+=issue_cycles;admissions+=issue_cycles
                ready=max(state[0],arrays[unit])
                for t in target:ready=max(ready,partial[t,column//bn])
                ready=accumulator(target,column,width,columns,ready,state,banks,
                    calendar,lengths,bank_width,counts,False,accumulator_base)
                programs[unit]=np.array([row,end,column,width,0,ready,ready,1],np.int64)
                admitted+=1
            chosen=-1;floor=np.iinfo(np.int64).max
            for unit in range(len(arrays)):
                p=programs[unit]
                if not p[VALID]:continue
                group=(p[CURSOR]//quantum)%2
                available=max(state[0],p[READY],latches[unit,group])
                if available<floor:chosen=unit;floor=available
            if chosen<0:break
            unit=chosen;p=programs[unit]
            k=p[CURSOR];depth=min(quantum,inner-k);group=(k//quantum)%2
            receivers=np.zeros(len(arrays),np.bool_);receivers[unit]=True
            if multicast:
                state[0]+=issue_cycles;admissions+=issue_cycles
                for other in range(len(arrays)):
                    q=programs[other]
                    if q[VALID] and q[ROW]==p[ROW] and q[END]==p[END] and q[CURSOR]==k:
                        receivers[other]=True
                        floor=max(floor,q[READY],latches[other,group])
            state[0]=max(state[0],floor)
            tags,masks=rectangle(starts[p[ROW]:p[END]]+k*source_stride,depth,source_stride,element_bytes)
            a=core.access(tags,masks,feature_directory,feature_slots,feature_links,
                feature_meta,state,pending,fb,fc,fl,priorities,victim_scan,counts,
                False,0,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
                SECTOR_BYTES,independent_drain,striped,separate_read_completion,
                feature_profile,dma_activity)
            source_row=p[ROW];source_end=p[END]
            saved+=(int(receivers.sum())-1)*(source_end-source_row)*depth*element_bytes
            for unit in range(len(arrays)):
                if not receivers[unit]:continue
                p=programs[unit]
                addresses=weight_bases[offset]+(p[COLUMN]+np.arange(p[WIDTH]))*wn+k*wk
                tags,masks=rectangle(addresses,depth,wk,element_bytes)
                b=core.access(tags,masks,weight_directory,weight_slots,weight_links,
                    weight_meta,state,pending,wb,wc,wl,priorities,victim_scan,counts,
                    False,0,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
                    SECTOR_BYTES,independent_drain,weight_striped,separate_read_completion,
                    weight_profile,dma_activity)
                # Each receiver has its own paid latch. The shared route does
                # not synchronize unrelated operands or change MAC work.
                begin=max(p[ISSUED],max(a,b)+1);consumed=begin+depth*precision
                latches[unit,group]=consumed;p[ISSUED]=consumed;p[CURSOR]+=depth
                occupied+=depth*precision
                if activity is not None:
                    activity[records]=np.array([begin,consumed,unit],np.int64);records+=1
                peak=max(peak,(p[END]-p[ROW]+p[WIDTH])*depth*element_bytes)
                if p[CURSOR]==inner:
                    done=consumed+bm+bn-2
                    target=targets[p[ROW]:p[END]]
                    stored=accumulator(target,p[COLUMN],p[WIDTH],columns,done,state,
                        banks,calendar,lengths,bank_width,counts,True,accumulator_base)
                    for t in target:partial[t,p[COLUMN]//bn]=stored
                    # A new output tile may not take the unique PE owner
                    # until the old tile's tail and final store both finish.
                    arrays[unit]=stored
                    completion=max(completion,stored);p[VALID]=0;tiles+=1
            ahead=k+prefetch_bytes//element_bytes
            if prefetch_bytes and (k*element_bytes)%DMA_BURST_BYTES==0 and ahead<inner:
                tags,masks=rectangle(starts[source_row:source_end]+ahead*source_stride,
                    min(quantum,inner-ahead),source_stride,element_bytes)
                core.access(tags,masks,feature_directory,feature_slots,feature_links,
                    feature_meta,state,pending,fb,fc,fl,priorities,victim_scan,counts,
                    False,0,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
                    SECTOR_BYTES,independent_drain,striped,separate_read_completion,
                    feature_profile,dma_activity,True)
                prefetches+=1
                for unit in range(len(arrays)):
                    if not receivers[unit]:continue
                    p=programs[unit]
                    addresses=weight_bases[offset]+(p[COLUMN]+np.arange(p[WIDTH]))*wn+ahead*wk
                    tags,masks=rectangle(addresses,min(quantum,inner-ahead),wk,element_bytes)
                    core.access(tags,masks,weight_directory,weight_slots,weight_links,
                        weight_meta,state,pending,wb,wc,wl,priorities,victim_scan,counts,
                        False,0,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
                        SECTOR_BYTES,independent_drain,weight_striped,separate_read_completion,
                        weight_profile,dma_activity,True)
                    prefetches+=1
        # Sparse offsets can repack target rows: the next offset reads the
        # actual previous partial, never an implied same-array accumulator.
    return completion,tiles,occupied,records,admissions,peak,prefetches,saved


@core.njit(cache=True)
def translated_read(tags,masks,saved_tags,saved_slots,slots,earliest,state,banks,
                    calendar,lengths,bank_width,striped,counts,profile):
    """Host representation of row-indexed slot fields, not a second CAM.

    The caller pays row translation-table reads. No directory client executes
    between final verification and this segment's last source-bank read.
    """
    done=earliest;view=slots.view(np.uint64)
    for at in range(len(tags)):
        where=np.searchsorted(saved_tags,tags[at]);slot=saved_slots[where]
        if slots[slot,0]!=tags[at] or masks[at]&~view[slot,1]:
            raise AssertionError('Translated owner lost before its last read')
        end,n=core.masked_transfer(slot,masks[at],max(earliest,slots[slot,5]),
            state,banks,calendar,lengths,bank_width,striped)
        slots[slot,3]=max(slots[slot,3],end);done=max(done,end)
        counts[6]+=n*SECTOR_BYTES
        if profile is not None:profile[6]+=n*SECTOR_BYTES
    return done


@core.njit(cache=True)
def translated_products(starts,targets,pointer,weight_bases,wk,wn,source_stride,inner,columns,
             output_rows,element_bytes,precision,bm,bn,arrays,latches,
             feature_directory,weight_directory,feature_slots,weight_slots,
             feature_links,weight_links,feature_meta,weight_meta,state,pending,
             banks,calendar,lengths,feature_ports,weight_ports,priorities,victim_scan,
             counts,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
             independent_drain,striped,weight_striped,separate_read_completion,
             accumulator_base,feature_profile,weight_profile,dma_activity,activity,
             issue_cycles,multicast,k_tile):
    """Closed cohorts retain one PE output owner per array through full K.

    Each K segment loads all real tags, verifies their final owners, then
    consumes through physical slots. Next-segment fill can overlap the old
    MAC tail once every old SRAM read has reserved its owner lifetime.
    """
    fb,fc,fl=feature_ports;wb,wc,wl=weight_ports
    quantum=SECTOR_BYTES//element_bytes
    partial=np.zeros((output_rows,(columns+bn-1)//bn),np.int64)
    initial=accumulator(np.arange(output_rows),0,columns,columns,state[0],
        state,banks,calendar,lengths,bank_width,counts,True,accumulator_base)
    partial[:,:]=initial
    completion=initial;tiles=0;occupied=0;records=0;admissions=0;peak=0;prefetches=0;saved=0;table_bytes=0
    ntiles=(columns+bn-1)//bn;table_ready=state[0]
    for offset in range(len(pointer)-1):
        total=((pointer[offset+1]-pointer[offset]+bm-1)//bm)*ntiles
        for batch in range(0,total,len(arrays)):
            programs=np.zeros((len(arrays),8),np.int64)
            for unit in range(min(len(arrays),total-batch)):
                task=batch+unit;row=pointer[offset]+(task//ntiles)*bm
                end=min(row+bm,pointer[offset+1]);column=(task%ntiles)*bn
                width=min(bn,columns-column);target=targets[row:end]
                state[0]+=issue_cycles;admissions+=issue_cycles
                ready=max(state[0],arrays[unit])
                for t in target:ready=max(ready,partial[t,column//bn])
                ready=accumulator(target,column,width,columns,ready,state,banks,
                    calendar,lengths,bank_width,counts,False,accumulator_base)
                programs[unit]=np.array([row,end,column,width,0,ready,ready,1],np.int64)
            for chunk in range(0,inner,k_tile):
                depth=min(k_tile,inner-chunk)
                a_starts=np.empty(len(arrays)*bm,np.int64);b_starts=np.empty(len(arrays)*bn,np.int64)
                na=nb=0;entries=0
                for p in programs:
                    if not p[VALID]:continue
                    for row in range(p[ROW],p[END]):
                        start=starts[row]+chunk*source_stride;a_starts[na]=start;na+=1
                        entries+=(start+(depth-1)*source_stride+element_bytes-1)//SLOT_BYTES-start//SLOT_BYTES+1
                    for column in range(p[COLUMN],p[COLUMN]+p[WIDTH]):
                        start=weight_bases[offset]+column*wn+chunk*wk;b_starts[nb]=start;nb+=1
                        entries+=(start+(depth-1)*wk+element_bytes-1)//SLOT_BYTES-start//SLOT_BYTES+1
                atags,amasks=rectangle(a_starts[:na],depth,source_stride,element_bytes)
                btags,bmasks=rectangle(b_starts[:nb],depth,wk,element_bytes)
                core.access(atags,amasks,feature_directory,feature_slots,feature_links,
                    feature_meta,state,pending,fb,fc,fl,priorities,victim_scan,counts,
                    False,0,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
                    SECTOR_BYTES,independent_drain,striped,separate_read_completion,
                    feature_profile,dma_activity,True)
                core.access(btags,bmasks,weight_directory,weight_slots,weight_links,
                    weight_meta,state,pending,wb,wc,wl,priorities,victim_scan,counts,
                    False,0,lookup_width,cam_latency,bank_width,burst_cycles,dram_latency,
                    SECTOR_BYTES,independent_drain,weight_striped,separate_read_completion,
                    weight_profile,dma_activity,True)
                # Paid final owner/valid verification, then row-indexed table
                # writes. Saved translations are captured only after all fill
                # admissions, including any old-slot replacements.
                verify=(len(atags)+len(btags)+lookup_width-1)//lookup_width+cam_latency
                write_cost=(entries*6+lookup_width-1)//lookup_width
                state[0]=max(state[0]+verify,table_ready)+write_cost
                admissions+=verify+write_cost;table_bytes+=entries*24
                aslots=np.empty(len(atags),np.int64);bslots=np.empty(len(btags),np.int64)
                for at in range(len(atags)):
                    if atags[at] not in feature_directory:raise AssertionError('Closed feature segment exceeds protected capacity')
                    aslots[at]=feature_directory[atags[at]]
                    if amasks[at]&~feature_slots.view(np.uint64)[aslots[at],1]:raise AssertionError('Incomplete translated feature fill')
                for at in range(len(btags)):
                    if btags[at] not in weight_directory:raise AssertionError('Closed weight segment exceeds protected capacity')
                    bslots[at]=weight_directory[btags[at]]
                    if bmasks[at]&~weight_slots.view(np.uint64)[bslots[at],1]:raise AssertionError('Incomplete translated weight fill')
                counts[3]+=len(atags)+len(btags)
                if feature_profile is not None:feature_profile[3]+=len(atags)
                if weight_profile is not None:weight_profile[3]+=len(btags)
                prefetches+=2
                while True:
                    chosen=-1;floor=np.iinfo(np.int64).max
                    for unit in range(len(arrays)):
                        p=programs[unit]
                        if not p[VALID] or p[CURSOR]>=chunk+depth:continue
                        group=(p[CURSOR]//quantum)%2
                        available=max(state[0],latches[unit,group])
                        if available<floor:chosen=unit;floor=available
                    if chosen<0:break
                    p=programs[chosen];k=p[CURSOR];take=min(quantum,chunk+depth-k);group=(k//quantum)%2
                    receivers=np.zeros(len(arrays),np.bool_);receivers[chosen]=True
                    if multicast:
                        for unit in range(len(arrays)):
                            q=programs[unit]
                            if q[VALID] and q[ROW]==p[ROW] and q[END]==p[END] and q[CURSOR]==k:
                                receivers[unit]=True;floor=max(floor,latches[unit,group])
                    # Only the slot field (64 bits) is read per active operand
                    # row; the row/column cursor selects its boundary entry.
                    reads=row_entries(starts[p[ROW]:p[END]]+k*source_stride,take,source_stride,element_bytes)
                    for unit in range(len(arrays)):
                        if receivers[unit]:
                            q=programs[unit]
                            addresses=weight_bases[offset]+(q[COLUMN]+np.arange(q[WIDTH]))*wn+k*wk
                            reads+=row_entries(addresses,take,wk,element_bytes)
                    service=(reads*2+lookup_width-1)//lookup_width
                    state[0]=max(state[0],floor)+service;admissions+=service;table_bytes+=reads*8
                    returned=state[0]+1
                    tags,masks=rectangle(starts[p[ROW]:p[END]]+k*source_stride,take,source_stride,element_bytes)
                    a=translated_read(tags,masks,atags,aslots,feature_slots,returned,state,fb,fc,fl,
                                      bank_width,striped,counts,feature_profile)
                    saved+=(int(receivers.sum())-1)*(p[END]-p[ROW])*take*element_bytes
                    for unit in range(len(arrays)):
                        if not receivers[unit]:continue
                        p=programs[unit]
                        addresses=weight_bases[offset]+(p[COLUMN]+np.arange(p[WIDTH]))*wn+k*wk
                        tags,masks=rectangle(addresses,take,wk,element_bytes)
                        b=translated_read(tags,masks,btags,bslots,weight_slots,returned,state,wb,wc,wl,
                                          bank_width,weight_striped,counts,weight_profile)
                        table_ready=max(table_ready,a+1,b+1)
                        begin=max(p[ISSUED],max(a,b)+1);consumed=begin+take*precision
                        latches[unit,group]=consumed;p[ISSUED]=consumed;p[CURSOR]+=take;occupied+=take*precision
                        if activity is not None:
                            activity[records]=np.array([begin,consumed,unit],np.int64);records+=1
                        peak=max(peak,(p[END]-p[ROW]+p[WIDTH])*take*element_bytes)
            for unit in range(len(arrays)):
                p=programs[unit]
                if not p[VALID]:continue
                done=p[ISSUED]+bm+bn-2
                target=targets[p[ROW]:p[END]]
                stored=accumulator(target,p[COLUMN],p[WIDTH],columns,done,state,
                    banks,calendar,lengths,bank_width,counts,True,accumulator_base)
                for t in target:partial[t,p[COLUMN]//bn]=stored
                arrays[unit]=stored;completion=max(completion,stored);tiles+=1
    # Translation SRAM is shared across invocations and diagnostic children.
    # Drain its last bank-return route before another caller can overwrite it;
    # already latched K words and the PE tail may continue independently.
    state[0]=max(state[0],table_ready)
    return completion,tiles,occupied,records,admissions,peak,prefetches,saved,table_bytes


def execute(sim,index,selected,columns=None,column=0):
    c=sim.c;g=sim.g;m=sim.memory;e=g.events[index];out=e['outputs'][0]
    source,weight=e['inputs'][:2]
    if e['kind']=='ancestor_merge':weight=e['inputs'][22]
    width=out['shape'][-1] if columns is None else columns
    inner=source['shape'][-1];element=out['element_bytes']
    _,_,budget=operand_capacity(c,element)
    if len(selected)*width*4>budget:raise ValueError('Direct accumulator exceeds output owner')
    starts,targets,pointer,bases,wk,wn=product_layout(g,index,selected)
    tiles=sum(math.ceil((pointer[i+1]-pointer[i])/c['array_rows']) for i in range(len(pointer)-1))*math.ceil(width/c['array_cols'])
    slices=(sum(math.ceil(min(c['k_tile'],inner-k)*element/SECTOR_BYTES) for k in range(0,inner,c['k_tile']))
            if c.get('direct_translated_tiles',False) else math.ceil(inner*element/SECTOR_BYTES))
    activity=sim.activity.matrix_buffer(tiles*slices) if sim.activity is not None else None
    which=int(weight['role']=='weight')
    banks,calendar,lengths=m.ports(2)
    translated=c.get('direct_translated_tiles',False)
    operation=translated_products if translated else products
    result=operation(starts,targets,pointer,bases+column*wn,
        wk,wn,source['stride'][-1]*source['element_bytes'],inner,width,len(selected),element,
        c['fp32_mac_cycles'] if element==4 else 1,c['array_rows'],c['array_cols'],
        sim.matrix,sim.input_latches,m.directories[0],m.directories[which],m.slots[0],m.slots[which],
        m.links[0],m.links[which],m.meta[0],m.meta[which],m.state,m.pending,banks,calendar,lengths,
        m.ports(0),m.ports(which),m.priorities,c['victim_scan_entries'],m.counts,
        c['cam_lookups_per_cycle'],c['cam_latency_cycles'],c['bank_bytes_per_cycle'],m.burst_cycles,
        c['dram_latency_cycles'],c.get('independent_eviction_drain',False),m.bank_layouts[0],
        m.bank_layouts[which],c.get('separate_read_completion',False),sim.accumulator_base(element),
        None if m.directory_counts is None else m.directory_counts[0],
        None if m.directory_counts is None else m.directory_counts[which],m.dma_activity,activity,
        c['issue_cycles'],c.get('direct_multicast',False),c['k_tile'] if translated else c.get('direct_prefetch_bytes',0))
    done,count,occupied,records,admissions,peak,prefetches,saved=result[:8]
    if translated:
        sim.stats['direct_translation_reserved_bytes']=translation_bytes(c)
        sim.stats['direct_translation_control_bytes']+=int(result[8])
    if activity is not None:sim.activity.add_matrix(activity[:records].copy(),done,tiles=int(count))
    sim.stats['direct_matrix_tiles']+=int(count)
    sim.stats['array_occupied_cycles']+=int(occupied)
    sim.stats['matrix_flops']+=g.flops(index,selected)*width//out['shape'][-1]
    sim.stats['direct_admission_cycles']+=int(admissions)
    sim.stats['direct_latch_peak_bytes']=max(sim.stats['direct_latch_peak_bytes'],int(peak))
    sim.stats['direct_latch_reserved_bytes']=input_latch_bytes(c)
    sim.stats['direct_descriptor_bytes']=descriptor_bytes(c)
    sim.stats['direct_prefetch_requests']+=int(prefetches)
    sim.stats['direct_multicast_saved_bytes']+=int(saved)
    return int(done)
