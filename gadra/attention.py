"""Bounded row softmax on the existing vector datapath and output banks."""
import math

from .protocol import SOFTMAX_PASSES
from .ready_queue import ReadyQueue


def pack_limit(config):
    """Independent row reductions supported by each finite vector context."""
    limit=config.get('attention_pack_limit',2)
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<1:
        raise ValueError('Attention pack limit must be a positive integer')
    return min(limit,config['vector_lanes']) if config.get('attention_pack_rows',False) else 1


def register_rows(config,sequence):
    """Partition existing lanes without splitting a reduction across rows."""
    return min(pack_limit(config),max(1,config['vector_lanes']//sequence))


def use_registers(config,sequence):
    """Choose an existing softmax path by its finite supply/service costs.

    This is an admission estimate, never the reported duration. The two
    paths still reserve every actual bank/vector/owner operation below.
    Short rows waste vector lanes in the row-at-a-time register path.
    """
    if not config.get('attention_register_softmax',False) or sequence>config['vector_lanes']:
        return False
    if not config.get('consumer_attention_batches_active',False):return True
    # Register softmax avoids three FP32 score passes through SRAM, but
    # pays one admission and SOFTMAX_PASSES vector beats for every row.
    bandwidth=config['sram_banks']*config['bank_bytes_per_cycle']
    lanes=config['vector_lanes']
    return register_rows(config,sequence)*sequence*(3*4*lanes+SOFTMAX_PASSES*bandwidth)>=(
        SOFTMAX_PASSES+config['issue_cycles'])*bandwidth*lanes


def register_bytes(config):
    # Two data/owner contexts. They replace, rather than supplement, this
    # many bytes of the configured control SRAM. No new vector lanes/ports.
    # Each context has one vector plus FP32 max/sum scalars and an owner.
    scalars=8*pack_limit(config)
    return 2*(config['vector_lanes']*4+scalars+ReadyQueue.entry_bytes) if config.get('attention_register_softmax',False) else 0


def staging_bytes(config):
    """Two rank-4 operand identities and seven owner/layout words per slot.

    Identity words: storage/version/address/element width, shape and stride.
    Validity, event/head/count/base/size/ready use seven further 64-bit words.
    Storage is rounded to the original control SRAM's sector granularity.
    """
    from .protocol import SECTOR_BYTES
    words=2*(4+4+4)+7
    return math.ceil(words*8/SECTOR_BYTES)*SECTOR_BYTES*config['output_buffers']


def reset_staging(sim):
    """A lease never crosses an operator/phase boundary or unknown writer."""
    if not sim.attention_staging_storage_bytes:return
    sim.attention_staging_leases=[None]*sim.c['output_buffers']
    sim.attention_staging_permission=0
    cost=math.ceil(sim.c['output_buffers']/sim.c['cam_lookups_per_cycle'])+1
    sim.memory.clock+=cost
    sim.stats['attention_coverage_staging_control_cycles']+=cost


def cohort_rows(sim,sequence,dim,element_bytes):
    """Maximum admitted query rows from the existing physical output owner."""
    available=sim.output_budget()-2*sequence*dim*element_bytes
    per_row=sequence*4+dim*(4+element_bytes)
    count=min(sim.c['array_rows']*sim.c.get('attention_query_tiles',1),available//per_row)
    if count>=sim.c['array_rows']:count=count//sim.c['array_rows']*sim.c['array_rows']
    if count<=0:raise ValueError('Attention staging exceeds output SRAM')
    return count


def query_capacity(sim,count,sequence,dim,element_bytes):
    if not (sim.attention_staging_storage_bytes and sim.c.get('coverage_attention_fixed_layout',False)):
        return count
    capacity=cohort_rows(sim,sequence,dim,element_bytes)
    if count>capacity:raise ValueError('Query count exceeds its fixed Attention layout')
    return capacity


def staging_key(sim,index,ids):
    from .graph import key
    from .coverage_rows import CoverageRows
    event=sim.g.events[index];out=event['outputs'][0]
    sequence=out['shape'][-2]
    capacity=query_capacity(sim,len(ids),sequence,out['shape'][-1],out['element_bytes'])
    return (index,int(ids[0])//sequence,capacity,sim.accumulator_base(4),
            tuple((key(s),CoverageRows.signature(s)) for s in event['inputs'][1:3]))


def reuse_staging(sim,index,ids):
    """Only a fully coverage-qualified K AND V can reuse a physical owner."""
    if not sim.attention_staging_storage_bytes:return False
    # Read only this selected output owner's finite descriptor. Field
    # comparisons run on existing control lanes, not an extra CAM/port.
    nbytes=sim.attention_staging_storage_bytes//sim.c['output_buffers']
    cost=math.ceil(nbytes/(4*sim.c['cam_lookups_per_cycle']))+sim.c['cam_latency_cycles']+1
    sim.memory.clock+=cost
    sim.stats['attention_coverage_staging_control_cycles']+=cost
    lease=sim.attention_staging_leases[sim.active_output_buffer]
    hit=(sim.attention_staging_permission==3 and lease is not None and
         lease[0]==staging_key(sim,index,ids))
    if hit:
        sim.memory.clock=max(sim.memory.clock,lease[3])
        sim.stats['attention_coverage_staging_hits']+=1
    return hit


def remember_staging(sim,index,ids,ready):
    if not sim.attention_staging_storage_bytes or sim.attention_staging_permission!=3:return
    out=sim.g.events[index]['outputs'][0];dim=out['shape'][-1];element=out['element_bytes']
    capacity=query_capacity(sim,len(ids),out['shape'][-2],dim,element)
    begin=sim.accumulator_base(4)+capacity*dim*element
    size=2*out['shape'][-2]*dim*element
    sim.attention_staging_leases[sim.active_output_buffer]=(staging_key(sim,index,ids),begin,size,ready)
    nbytes=sim.attention_staging_storage_bytes//sim.c['output_buffers']
    cost=math.ceil(nbytes/(4*sim.c['cam_lookups_per_cycle']))+1
    sim.memory.clock+=cost
    sim.stats['attention_coverage_staging_control_cycles']+=cost


def scratch_write(sim,nbytes,earliest,offset,qualified=False):
    """Only statically bounded score/probability/AV opcodes bypass lookup.

    The pipeline admission fixes their ranges beyond the K/V payload;
    probabilities fit within FP32 scores. Payload fills and unknown writers
    always use the ordinary invalidating path.
    """
    if qualified and sim.attention_staging_storage_bytes:
        return sim.output_transfer(nbytes,earliest,write=True,offset=offset,staging_disjoint=True)
    return sim.output_transfer(nbytes,earliest,write=True,offset=offset)


def set_output_layout(sim,enabled):
    """Repartition only after every old physical output/operand owner drains.

    The layout bit is part of the existing Attention controller phase. No
    capacity or port is added; ordinary feeders cannot run in this layout.
    """
    if not sim.c.get('consumer_attention_reclaim_active',False):return
    if sim.attention_output_layout_active==enabled:return
    m=sim.memory
    ready=max(m.clock,sim.cycle,sim.vector,int(sim.matrix.max()),
              int(sim.buffer_ready.max()),int(sim.feeder_ready.max()),
              int(sim.feeder_b_ready.max()),int(sim.copy_ready.max()),
              int(sim.input_latches.max()),int(sim.softmax_ready.max()),
              int(sim.attention_pipeline_ready.max()),int(m.ports(2)[0].max()))
    sim.stats['attention_layout_wait_cycles']+=ready-m.clock
    # Invalidate any old K/V lease before changing physical addresses. The
    # ordinary feeder's operand identities are local to each products call.
    m.clock=ready
    reset_staging(sim)
    m.clock+=sim.c['issue_cycles']
    sim.cycle=max(sim.cycle,m.clock)
    sim.stats['attention_layout_switch_cycles']+=sim.c['issue_cycles']
    sim.stats['attention_layout_switches']+=1
    sim.attention_output_layout_active=enabled
    if enabled:
        from .feeder import operand_capacity
        sim.stats['attention_layout_reclaimed_bytes']=max(
            sim.stats['attention_layout_reclaimed_bytes'],
            sim.c['output_sram_bytes']-sim.c['output_buffers']*
            operand_capacity(sim.c,4)[2])


def pipeline(sim,count,sequence,dim,element_bytes,earliest,*,reuse_kv=False):
    """Interleave QK/normalization/AV subtiles within one real staging owner.

    A query descriptor remains owned through its AV read/cast and release.
    The output buffer separately retains all K/V and results until its final
    ordinary store. No K/V read is replaced with a free multicast.
    """
    c=sim.c;m=sim.memory;bm=c['array_rows'];bn=c['array_cols']
    if math.ceil(count/bm)>len(sim.attention_pipeline_ready):
        raise ValueError('Attention query cohort exceeds finite descriptor count')
    precision=c['fp32_mac_cycles'] if element_bytes==4 else 1
    qbytes=count*dim*element_bytes;kvbytes=sequence*dim*element_bytes
    capacity=query_capacity(sim,count,sequence,dim,element_bytes)
    reserved_qbytes=capacity*dim*element_bytes
    payload=reserved_qbytes+2*kvbytes;score_bytes=count*sequence*4;av_bytes=count*dim*4
    staging=payload+score_bytes+av_bytes
    if staging>sim.output_budget():raise ValueError('Attention staging exceeds output owner')
    disjoint=sim.attention_staging_storage_bytes and element_bytes<=4
    if disjoint:
        # One finite layout check for the complete microprogram, rather than
        # rereading owner ranges for every softmax row. All writes below are
        # bounded by score_bytes/av_bytes and their original loop limits.
        cost=sim.c['cam_latency_cycles']+math.ceil(7/sim.c['cam_lookups_per_cycle'])
        earliest=max(earliest,sim.memory.clock)+cost
        sim.memory.clock+=cost
        sim.stats['attention_coverage_staging_control_cycles']+=cost
    sim.stats['attention_peak_staging_bytes']=max(sim.stats['attention_peak_staging_bytes'],staging)
    scratch=sim.accumulator_base(4);score_base=scratch+payload;av_base=score_base+score_bytes
    # A single array wave already has no spare query cohort to prepare.
    # Keep the original interleave there; larger admitted cohorts can hide
    # their later QK preparation behind the normalization/AV consumers.
    frontload=(c.get('consumer_attention_qk_frontload_active',False) and
               math.ceil(count/bm)>c['arrays'])
    def dispatch(ready):
        # A later QK can block finite dispatch and retire bank reservations.
        # A previously completed QK never permits issuing back into that past.
        return max(ready,m.clock) if frontload else ready
    if capacity!=count:
        # Padding reserves addresses only. Write actual Q and K/V in their
        # disjoint ranges; the second real copy opcode pays an issue stage.
        loaded=sim.output_transfer(qbytes,dispatch(earliest),write=True,offset=scratch)
        if not reuse_kv:
            loaded=sim.output_transfer(2*kvbytes,dispatch(loaded+c['issue_cycles']),
                write=True,offset=scratch+reserved_qbytes)
            sim.stats['attention_fixed_layout_copy_cycles']+=c['issue_cycles']
    else:
        loaded=sim.output_transfer(qbytes if reuse_kv else payload,dispatch(earliest),write=True,offset=scratch)
    if reuse_kv:sim.stats['attention_coverage_staging_saved_write_bytes']+=2*kvbytes
    completion=loaded
    def prepare_queries():
        for ordinal,begin in enumerate(range(0,count,bm)):
            n=min(bm,count-begin)
            # The frontloaded cohort owns distinct existing descriptors until
            # AV retires. Fixed ordinals avoid reacquiring a QK-only owner.
            slot=ordinal if frontload else int(sim.attention_pipeline_ready.argmin())
            m.clock=max(m.clock,int(sim.attention_pipeline_ready[slot]))+c['issue_cycles']
            ready=max(loaded,m.clock)
            for _ in range(math.ceil(sequence/bn)):
                ready=sim.output_transfer(n*dim*element_bytes,dispatch(ready),
                                          offset=scratch+begin*dim*element_bytes)
            ready=sim.output_transfer(kvbytes,dispatch(ready),offset=scratch+reserved_qbytes)
            qk_count=math.ceil(sequence/bn)
            qk_init=dim*precision if c['array_wave_pipeline'] else 0
            qk_duration=dim*precision+bm+bn-2
            done=sim.issue_matrix(qk_count,qk_duration,dispatch(ready),qk_init)
            sim.stats['array_occupied_cycles']+=qk_count*(qk_init or qk_duration)
            score=score_base+begin*sequence*4
            done=scratch_write(sim,n*sequence*4,dispatch(done+c['issue_cycles']),score,disjoint)
            yield begin,n,slot,done,score
    pending=prepare_queries()
    if frontload:
        # These host tuples represent the already paid 24B query descriptors:
        # row/cursor, phase/valid and completed-QK time. Their scores remain in
        # this output owner's original bounded, disjoint score ranges.
        pending=list(pending)
        m.clock+=c['issue_cycles']
        sim.stats['attention_qk_frontload_cohorts']+=1
        sim.stats['attention_pipeline_control_cycles']+=c['issue_cycles']
    for begin,n,slot,done,score in pending:
        probabilities=n*sequence*element_bytes
        if use_registers(c,sequence):
            done=normalize(sim,n,sequence,element_bytes,score,dispatch(done),staging_disjoint=disjoint,
                           dispatch_floor=frontload)
            sim.stats['attention_scratch_bytes']+=n*sequence*4*2+probabilities*2
        else:
            for _ in range(3):done=sim.output_transfer(n*sequence*4,dispatch(done),offset=score)
            done=scratch_write(sim,n*sequence*4,dispatch(done),score,disjoint)
            cost=math.ceil(n*sequence*SOFTMAX_PASSES/c['vector_lanes'])
            done=sim.issue_vector(dispatch(done),cost);sim.stats['vector_cycles']+=cost
            done=scratch_write(sim,probabilities,dispatch(done),score,disjoint)
            sim.stats['attention_scratch_bytes']+=n*sequence*4*5+probabilities*2
        done=sim.output_transfer(probabilities,dispatch(done),offset=score)
        done=sim.output_transfer(kvbytes,dispatch(done),offset=scratch+reserved_qbytes+kvbytes)
        av_count=math.ceil(dim/bn)
        av_init=sequence*precision if c['array_wave_pipeline'] else 0
        av_duration=sequence*precision+bm+bn-2
        done=sim.issue_matrix(av_count,av_duration,dispatch(done+c['issue_cycles']),av_init)
        sim.stats['array_occupied_cycles']+=av_count*(av_init or av_duration)
        av=av_base+begin*dim*4
        done=scratch_write(sim,n*dim*4,dispatch(done),av,disjoint)
        done=sim.output_transfer(n*dim*4,dispatch(done),offset=av)
        cost=math.ceil(n*dim/c['vector_lanes'])
        done=sim.issue_vector(dispatch(done),cost)+c['issue_cycles']
        sim.stats['vector_cycles']+=cost
        sim.attention_pipeline_ready[slot]=done;completion=max(completion,done)
        sim.stats['attention_pipeline_query_tiles']+=1
        sim.stats['attention_pipeline_control_cycles']+=4*c['issue_cycles']
    sim.stats['attention_pipeline_descriptor_bytes']=sim.attention_pipeline_storage_bytes
    return int(completion)


def normalize(sim,count,sequence,element_bytes,score_base,earliest,*,staging_disjoint=False,dispatch_floor=False):
    """Read each score once; keep its max/exp/sum/normalize state in registers.

    Packed probability writes may overlap older FP32 score rows. The loaded
    prefix completion gates every store, so neither delayed bank reads nor
    out-of-order vector reservations can lose an unread score.
    """
    if sequence>sim.c['vector_lanes']:raise ValueError('Softmax row exceeds register context')
    m=sim.memory
    completed=earliest;loaded_prefix=earliest
    batch=register_rows(sim.c,sequence)
    for row in range(0,count,batch):
        n=min(batch,count-row)
        slot=int(sim.softmax_ready.argmin())
        m.clock=max(m.clock,int(sim.softmax_ready[slot]))+sim.c['issue_cycles']
        loaded=sim.output_transfer(n*sequence*4,max(earliest,m.clock),offset=score_base+row*sequence*4)
        loaded_prefix=max(loaded_prefix,loaded)
        done=sim.issue_vector(loaded,SOFTMAX_PASSES)
        stored=scratch_write(sim,n*sequence*element_bytes,max(done,loaded_prefix,m.clock if dispatch_floor else 0),
                             score_base+row*sequence*element_bytes,staging_disjoint)
        sim.softmax_ready[slot]=stored;completed=max(completed,stored)
        sim.stats['vector_cycles']+=SOFTMAX_PASSES
        sim.stats['attention_register_admission_cycles']+=sim.c['issue_cycles']
        sim.stats['attention_register_internal_read_bytes']+=n*sequence*4*SOFTMAX_PASSES
        sim.stats['attention_register_internal_write_bytes']+=n*sequence*4*4
        if n>1:sim.stats['attention_packed_softmax_programs']+=1
    sim.stats['attention_register_softmax_rows']+=count
    sim.stats['attention_softmax_register_bytes']=sim.softmax_register_storage_bytes
    return int(completed)
