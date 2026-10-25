"""Lower actual strided tile bytes to bounded SRAM-to-SRAM routing beats.

The edge arrays are host lowering, like the existing sparse offset lists.
They do not store a payload or authorize a read before directory/operand ready.
"""
import math
import numpy as np

from . import memory_core as core
from .protocol import SECTOR_BYTES,SLOT_BYTES,SLOT_SECTORS


def descriptor_register_bits(config):
    """Compact fields, pipelined mask tree and next-set-bit controller."""
    count=config.get('consumer_copy_batch_active',0)
    if not count or not config.get('consumer_copy_register_control',False):return 0
    slots=max(min(config['cam_entries'],config['feature_sram_bytes']//SLOT_BYTES),
              config['weight_sram_bytes']//SLOT_BYTES)
    slot_bits=max(1,(slots-1).bit_length())
    # Tag, physical slot, sector mask and valid; each OR node has a valid bit.
    fields=count*(64+slot_bits+SLOT_SECTORS+1)
    if config.get('consumer_adjacent_supply_active',False):
        # Source role remains feature/weight. A separate destination selector
        # identifies either of the original feeder half-slots.
        halfslots=config['arrays']*config.get('operand_buffers_per_array',2)
        fields+=count*max(1,(halfslots-1).bit_length())
    tree=(count-1)*(SLOT_SECTORS+1)
    if config.get('consumer_copy_source_credit',False):
        # After the initial mask reduction, reuse those pipeline registers
        # for each descriptor's next-sector, decoded bank/lane and ready bit.
        selector=(SLOT_SECTORS-1).bit_length()+max(1,(config['sram_banks']-1).bit_length())
        selector+=max(1,(config.get('copy_register_groups',1)-1).bit_length())+1
        tree=max(tree,count*selector)
    # Remaining mask, sector selector, and occupancy count.
    controller=SLOT_SECTORS+(SLOT_SECTORS-1).bit_length()+max(1,count.bit_length())
    if config.get('consumer_copy_mask_pipeline_active',False):
        # One selected sector and valid between mask selection and fanout.
        controller+=(SLOT_SECTORS-1).bit_length()+1
    # Outstanding source sectors, source-directory selector and issue-done.
    outstanding=count*(SLOT_SECTORS.bit_length()+2) if config.get('consumer_copy_pipeline',False) else 0
    if config.get('consumer_copy_source_defer',False):
        # At most one queued source request per descriptor: valid + sector.
        outstanding+=count*(1+(SLOT_SECTORS-1).bit_length())
    return fields+tree+controller+outstanding


def register_bytes(config):
    if not config.get('bounded_operand_copy',False):return 0
    # A sector payload, two 64-bit route/address-control words and a busy bit
    # per lane. Feature/weight share lanes; none are private to tile/array.
    lanes=config['sram_banks']*config.get('copy_register_groups',1)
    return lanes*(SECTOR_BYTES+2*np.dtype(np.int64).itemsize)+math.ceil(lanes/(8*SECTOR_BYTES))*SECTOR_BYTES


@core.njit(cache=True)
def scratch_transfer(base,count,column,width,pitch,element_bytes,earliest,state,
                     banks,calendar,lengths,bank_width,stats,write):
    """Coalesced strided scratch access by one bounded vector register batch."""
    span=(width*element_bytes+2*SECTOR_BYTES-2)//SECTOR_BYTES
    sectors=np.empty(count*span,np.int64);used=0
    for row in range(count):
        begin=base+row*pitch+column*element_bytes
        for sector in range(begin//SECTOR_BYTES,(begin+width*element_bytes-1)//SECTOR_BYTES+1):
            sectors[used]=sector;used+=1
    demand=np.bincount(np.unique(sectors[:used])%len(banks),minlength=len(banks))
    done=earliest
    for bank in range(len(banks)):
        nbytes=demand[bank]*SECTOR_BYTES
        if not nbytes:continue
        end=core.reserve_bank(bank,earliest,(nbytes+bank_width-1)//bank_width,state[0],calendar,lengths)
        banks[bank]=max(banks[bank],end);done=max(done,end)
        stats[7 if write else 6]+=nbytes
    return done,int(demand.sum())*SECTOR_BYTES


@core.njit(cache=True)
def mapping(starts,columns,source_stride,element_bytes,destination,pitch,transpose=False):
    """Exact source-sector -> destination-sector edges, including byte tails.

    A source sector can feed many gathered rows; transposed weights can scatter
    into many destination sectors. Each distinct transfer is charged once.
    Writes retain the real byte enables, so untouched operand bytes survive.
    """
    maximum=len(starts)*columns*4
    source=np.empty(maximum,np.int64);target=np.empty(maximum,np.int64);used=0
    for row in range(len(starts)):
        column=0
        while column<columns:
            address=starts[row]+column*source_stride
            dest=destination+(column*pitch+row*element_bytes if transpose else row*pitch+column*element_bytes)
            first=address//SECTOR_BYTES;last=(address+element_bytes-1)//SECTOR_BYTES
            out_first=dest//SECTOR_BYTES;out_last=(dest+element_bytes-1)//SECTOR_BYTES
            for sector in range(first,last+1):
                for output in range(out_first,out_last+1):
                    lo=max(0,sector*SECTOR_BYTES-address,output*SECTOR_BYTES-dest)
                    hi=min(element_bytes,(sector+1)*SECTOR_BYTES-address,(output+1)*SECTOR_BYTES-dest)
                    if lo>=hi:continue
                    source[used]=sector;target[used]=output;used+=1
            step=1
            if not transpose and source_stride==element_bytes:
                step=min(columns-column,(SECTOR_BYTES-address%SECTOR_BYTES)//element_bytes,
                         (SECTOR_BYTES-dest%SECTOR_BYTES)//element_bytes)
                step=max(1,step)
            column+=step
    source=source[:used];target=target[:used]
    keys=np.unique(source)
    # Compact row IDs bound this encoding by the finite tile size and physical
    # output address span; virtual source addresses are never shifted/packed.
    low=target.min();span=target.max()-low+1
    edges=np.unique(np.searchsorted(keys,source)*span+target-low)
    groups=edges//span
    counts=np.bincount(groups,minlength=len(keys))
    pointer=np.empty(len(keys)+1,np.int64);pointer[0]=0
    pointer[1:]=np.cumsum(counts)
    return keys,pointer,edges%span+low


@core.njit(cache=True)
def byte_controls(starts,columns,source_stride,element_bytes,destination,pitch,transpose=False):
    """Functional lowering of each emitted edge's byte enables and selectors.

    Hardware derives these from the current tile's strided addresses, not a
    persistent table of the whole graph. Host controls only verify that the
    traffic edges can transport exactly the actual bytes, including aliases.
    A selector of -1 disables that destination byte; 0..31 selects a lane.
    """
    keys,pointer,targets=mapping(starts,columns,source_stride,element_bytes,destination,pitch,transpose)
    select=np.full((len(targets),SECTOR_BYTES),-1,np.int8)
    for row in range(len(starts)):
        for column in range(columns):
            source=starts[row]+column*source_stride
            dest=destination+(column*pitch+row*element_bytes if transpose else row*pitch+column*element_bytes)
            for byte in range(element_bytes):
                key=(source+byte)//SECTOR_BYTES;target=(dest+byte)//SECTOR_BYTES
                at=np.searchsorted(keys,key)
                edge=pointer[at]+np.searchsorted(targets[pointer[at]:pointer[at+1]],target)
                select[edge,(dest+byte)%SECTOR_BYTES]=(source+byte)%SECTOR_BYTES
    return keys,pointer,targets,select
