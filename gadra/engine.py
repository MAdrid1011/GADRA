"""Finite-port, finite-capacity transaction/tile cycle model."""
from collections import OrderedDict,Counter
import heapq
import math

import numpy as np

from .graph import rows,key,coverage_layout
from .diagnostics import equivalent_count
from .predictor import uses_witness_selector
from .protocol import (SECTOR_BYTES,SLOT_BYTES,SLOT_SECTORS,NORM_PASSES,DIAGNOSTIC_FORK_DESCRIPTOR_BYTES,
                       GELU_PASSES,VECTOR_NORM_PASSES,SOFTMAX_PASSES,DMA_BURST_BYTES,BURST_SECTORS,DRAM_TIME_TICKS)


class ReferenceMemory:
    sector_bytes=SECTOR_BYTES
    slot_bytes=SLOT_BYTES

    def __init__(self,config,coverage):
        self.c=config;self.coverage=coverage
        self.caches=[OrderedDict(),OrderedDict()]
        self.physical_slots=[{},{}]
        self.capacities=[min(config['cam_entries'],config['feature_sram_bytes']//self.slot_bytes),config['weight_sram_bytes']//self.slot_bytes]
        self.clock=0;self.bus=0
        self.pending=[0]*config['dma_outstanding'];heapq.heapify(self.pending)
        self.banks=np.zeros(config['sram_banks'],dtype=np.int64)
        self.calendar=np.zeros((config['sram_banks'],config['queue_entries'],2),dtype=np.int64)
        self.bank_lengths=np.zeros(config['sram_banks'],dtype=np.int64)
        self.stats=Counter()

    def dma(self,earliest,write=False):
        free=heapq.heappop(self.pending)
        issued=max(earliest,free)
        begin=max(issued,self.bus)
        self.bus=begin+DMA_BURST_BYTES*self.c['clock_hz']/self.c['dram_bandwidth_bytes_per_second']
        finish=math.ceil(self.bus)+self.c['dram_latency_cycles']
        heapq.heappush(self.pending,finish)
        self.stats['dram_write_bytes' if write else 'dram_read_bytes']+=DMA_BURST_BYTES
        self.stats['dma_transactions']+=1
        return finish

    def access(self,sectors,weight=False,write=False,ready=0):
        if not len(sectors):return self.clock
        cache=self.caches[int(weight)];capacity=self.capacities[int(weight)]
        physical_slots=self.physical_slots[int(weight)]
        blocks=sectors//(self.slot_bytes//self.sector_bytes)
        tags,first=np.unique(blocks,return_index=True)
        masks=np.zeros(len(tags),dtype=np.uint64)
        group=np.searchsorted(tags,blocks)
        np.bitwise_or.at(masks,group,np.left_shift(np.uint64(1),(sectors%(self.slot_bytes//self.sector_bytes)).astype(np.uint64)))
        completion=self.clock
        self.clock+=math.ceil(len(tags)/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
        for tag,mask in zip(tags.tolist(),masks.tolist()):
            if tag in cache:
                valid,dirty,available=cache.pop(tag)
                slot=physical_slots[tag]
                self.stats['directory_hits']+=1
            else:
                valid=dirty=0;available=0
                self.stats['directory_misses']+=1
                if len(cache)>=capacity:
                    old_tag,(old_valid,old_dirty,old_available)=cache.popitem(last=False)
                    slot=physical_slots.pop(old_tag)
                    # A producing task owns its slot until completion. Late
                    # completion can never modify a newly allocated owner.
                    self.clock=max(self.clock,old_available)
                    for burst in range(self.slot_bytes//DMA_BURST_BYTES):
                        if old_dirty&(((1<<BURST_SECTORS)-1)<<(burst*BURST_SECTORS)):self.dma(self.clock,write=True)
                    self.stats['evictions']+=1
                else:
                    slot=len(cache)
                physical_slots[tag]=slot
            missing=mask&~valid
            if not write:
                for burst in range(self.slot_bytes//DMA_BURST_BYTES):
                    bits=((1<<BURST_SECTORS)-1)<<(burst*BURST_SECTORS)
                    if missing&bits:available=max(available,self.dma(self.clock));valid|=bits
            else:
                valid|=mask;dirty|=mask;available=max(available,ready)
            # Bank-level byte demand is accounted separately from DRAM miss
            # bytes; repeated cache hits still consume SRAM read service.
            bank=slot%self.c['sram_banks']
            nbytes=mask.bit_count()*self.sector_bytes
            bank_begin=max(self.clock,int(self.banks[bank]),available if not write else ready)
            bank_end=bank_begin+math.ceil(nbytes/self.c['bank_bytes_per_cycle'])
            self.banks[bank]=bank_end
            completion=max(completion,bank_end)
            cache[tag]=(valid,dirty,available)
        self.stats['sram_write_bytes' if write else 'sram_read_bytes']+=len(sectors)*self.sector_bytes
        return completion

    def finish(self):
        ready=max([self.clock,*self.banks.tolist(),*self.pending])
        for cache in self.caches:
            for valid,dirty,available in cache.values():
                for burst in range(self.slot_bytes//DMA_BURST_BYTES):
                    if dirty&(((1<<BURST_SECTORS)-1)<<(burst*BURST_SECTORS)):ready=max(ready,self.dma(max(ready,available),write=True))
        return max(ready,*self.pending)


class Memory:
    sector_bytes=SECTOR_BYTES
    slot_bytes=SLOT_BYTES

    def __init__(self,config,coverage):
        from . import memory_core as core
        self.core=core;self.c=config
        if config.get('slot_skewed_banks',False) and config.get('sector_striped_banks',False):
            raise ValueError('Select one physical directory bank layout')
        self.bank_layout=2 if config.get('slot_skewed_banks',False) else int(config.get('sector_striped_banks',False))
        weight_layout=config.get('weight_bank_layout',self.bank_layout)
        if not isinstance(weight_layout,int) or weight_layout not in (0,1,2):
            raise ValueError('Weight bank layout must be slot(0), sector(1), or skew(2)')
        self.bank_layouts=(self.bank_layout,weight_layout)
        self.capacities=[min(config['cam_entries'],config['feature_sram_bytes']//self.slot_bytes),config['weight_sram_bytes']//self.slot_bytes]
        self.directories=[core.new_directory(),core.new_directory()]
        self.slots=[np.zeros((n,6),dtype=np.int64) for n in self.capacities]
        self.links=[np.full((n,2),-1,dtype=np.int64) for n in self.capacities]
        self.meta=[np.array([0,-1,-1],dtype=np.int64) for _ in self.capacities]
        # Additional observer words never enter simulated state decisions.
        self.state=np.zeros(6 if config.get('observe_copy_admission',False) else 2,dtype=np.int64)
        self.pending=np.zeros(config['dma_outstanding'],dtype=np.int64)
        groups=3 if config.get('independent_sram_ports',False) else 1
        self.banks=np.zeros(groups*config['sram_banks'],dtype=np.int64)
        self.calendar=np.zeros((groups*config['sram_banks'],config['queue_entries'],2),dtype=np.int64)
        self.bank_lengths=np.zeros(groups*config['sram_banks'],dtype=np.int64)
        self.counts=np.zeros(8,dtype=np.int64)
        # Host-side observation only: no readiness, allocation or issue input.
        self.directory_counts=np.zeros((2,8),dtype=np.int64) if config.get('observe_memory_paths',False) else None
        self.dma_activity=core.new_intervals() if config.get('observe_execution_intervals',False) else None
        self.priorities=core.new_directory()
        self.stats=Counter()
        self.burst_cycles=math.ceil(DMA_BURST_BYTES*config['clock_hz']*DRAM_TIME_TICKS/config['dram_bandwidth_bytes_per_second'])

    @property
    def clock(self):return int(self.state[0])

    @clock.setter
    def clock(self,value):self.state[0]=value

    def ports(self,group):
        """Physical feature(0), weight(1), output/operand(2) SRAM banks."""
        if not self.c.get('independent_sram_ports',False):return self.banks,self.calendar,self.bank_lengths
        width=self.c['sram_banks'];selected=slice(group*width,(group+1)*width)
        return self.banks[selected],self.calendar[selected],self.bank_lengths[selected]

    def update_stats(self):
        names=('dram_read_bytes','dram_write_bytes','dma_transactions','directory_hits',
               'directory_misses','evictions','sram_read_bytes','sram_write_bytes')
        self.stats.update({name:int(value)-self.stats[name] for name,value in zip(names,self.counts)})

    def observed_copy_admission(self):
        """Copy-only shared dispatch advances; not additive E2E stall time."""
        if len(self.state)==2:return None
        return dict(source_queue_advances=int(self.state[2]),
                    source_dispatch_advance_cycles=int(self.state[3]),
                    target_queue_advances=int(self.state[4]),
                    target_dispatch_advance_cycles=int(self.state[5]),
                    scope='Observed shared copy-dispatch frontier advances; not E2E removable latency')

    def access(self,sectors,weight=False,write=False,ready=0,prefetch=False,copy_sink=None):
        """Lookup/fill now; ready gates a producer write or consumer read."""
        if not len(sectors):return self.clock
        blocks=sectors//SLOT_SECTORS
        tags=np.unique(blocks).astype(np.int64)
        masks=np.zeros(len(tags),dtype=np.uint64)
        np.bitwise_or.at(masks,np.searchsorted(tags,blocks),np.left_shift(np.uint64(1),(sectors%SLOT_SECTORS).astype(np.uint64)))
        which=int(weight)
        banks,calendar,lengths=self.ports(which)
        completion=self.core.access(tags,masks,self.directories[which],self.slots[which],
            self.links[which],self.meta[which],self.state,self.pending,banks,calendar,lengths,self.priorities,self.c['victim_scan_entries'],self.counts,
            write,ready,self.c['cam_lookups_per_cycle'],self.c['cam_latency_cycles'],
            self.c['bank_bytes_per_cycle'],self.burst_cycles,self.c['dram_latency_cycles'],self.sector_bytes,
            self.c.get('independent_eviction_drain',False),self.bank_layouts[which],
            self.c.get('separate_read_completion',False),
            None if self.directory_counts is None else self.directory_counts[which],self.dma_activity,prefetch,copy_sink)
        return int(completion)

    def finish(self):
        self.clock=max(self.clock,int(self.banks.max()))
        ready=self.clock
        for which,(slots,meta) in enumerate(zip(self.slots,self.meta)):
            banks,calendar,lengths=self.ports(which)
            ready=max(ready,int(self.core.flush(slots,meta,self.state,self.pending,self.counts,
                self.burst_cycles,self.c['dram_latency_cycles'],self.sector_bytes,
                banks,calendar,lengths,self.c['bank_bytes_per_cycle'],self.bank_layouts[which],
                None if self.directory_counts is None else self.directory_counts[which],self.dma_activity)))
        self.update_stats()
        return ready

    def observed_paths(self):
        """Partition actual transactions, including dirty eviction/final drain."""
        if self.directory_counts is None:return None
        names=('dram_read_bytes','dram_write_bytes','dma_transactions','directory_hits',
               'directory_misses','evictions','sram_read_bytes','sram_write_bytes')
        direct=self.counts-self.directory_counts.sum(axis=0)
        if np.any(direct<0):raise AssertionError('A memory transaction was observed twice')
        return {name:dict(zip(names,map(int,values))) for name,values in zip(
            ('feature_directory','weight_directory','output_and_direct_dma'),
            (*self.directory_counts,direct))}

    def discard(self,sectors,hold_until):
        if not len(sectors):return 0
        tags=np.unique(sectors//SLOT_SECTORS).astype(np.int64);masks=np.zeros(len(tags),dtype=np.uint64)
        np.bitwise_or.at(masks,np.searchsorted(tags,sectors//SLOT_SECTORS),np.left_shift(np.uint64(1),(sectors%SLOT_SECTORS).astype(np.uint64)))
        return int(self.core.discard(tags,masks,self.directories[0],self.slots[0],hold_until))

    def resident(self,sectors):
        return int(self.core.resident(sectors,self.directories[0],self.slots[0]))

    def protect(self,sectors):
        self.priorities=self.core.new_directory()
        if len(sectors):
            tags,counts=np.unique(sectors//SLOT_SECTORS,return_counts=True)
            for tag,count in zip(tags,counts):self.priorities[int(tag)]=int(count)
        self.core.refresh_priority(self.slots[0],self.meta[0],self.priorities)


class Simulator:
    @staticmethod
    def can_reclaim(event,spec,same_allocation=None):
        from .normalization_timing import current_statistics
        if 'diagnostic_control_rows' in event:return False
        if event.get('indexed_source',False) and spec is event['inputs'][0]:return False
        local_input={'progressive_difference':0,'progressive_update':1,'progressive_observe':0}.get(event['kind'])
        if local_input is not None:
            # Only the local contribution has an injective consumer map.
            # Parent/seed/context and global persistent state may be reused
            # by later local rows; the surrounding last-use/carry/alias and
            # protected-storage guards still apply before discarding bytes.
            source=event['inputs'][local_input]
            same=same_allocation or (lambda a,b:a['storage']==b['storage'])
            return (spec is source and spec['contiguous'] and len(spec['shape'])==2
                    and all(not same(other,spec) for i,other in enumerate(event['inputs']) if i!=local_input))
        row_local=event['kind']=='linear' or event.get('op','').startswith((
            'aten.add.','aten.add_.','aten.sub.','aten.sub_.','aten.mul.','aten.mul_.',
            'aten.div.','aten.div_.','aten.relu',
            'aten.gelu','aten.layer_norm','aten.to.','aten.copy_','aten.clamp_min','aten.linalg_vector_norm')) or current_statistics(event)
        if not row_local or not spec['contiguous'] or rows(spec)!=rows(event['outputs'][0]):return False
        # Shifted aliases can consume this row again in another output tile.
        return all((other['address'],other['shape'],other['stride'])==(spec['address'],spec['shape'],spec['stride'])
                   for other in event['inputs'] if key(other)==key(spec))

    def __init__(self,graph,config,coverage=False,order=None,demand=False,coverage_links=None):
        pilot_row_chain_limit=config.get('row_chain_max_matrices')
        if coverage and config.get('coverage_compact_references',False) and not config.get('coverage_linked_supply',False):
            raise ValueError('Compact coverage references require witnessed linked supply')
        if coverage and config.get('coverage_link_incremental_pairs',False) and not config.get('coverage_linked_supply',False):
            raise ValueError('Incremental coverage pairs require witnessed linked supply')
        if coverage and config.get('coverage_attention_links',False) and not config.get('coverage_linked_supply',False):
            raise ValueError('Attention coverage requires witnessed linked supply')
        if coverage and config.get('coverage_linked_supply',False):
            if (coverage_links is None or order is None or
                not config.get('coverage_exact_rows',False)):
                raise ValueError('Linked supply requires real coverage links and exact native row provenance')
            config=dict(config,coverage_shared_boundary=True)
        if demand and config.get('demand_local_completion',False):
            if not config.get('finite_diagnostic_pipeline',False):
                raise ValueError('Demand completion domains require the finite diagnostic pipeline')
            config=dict(config,event_local_completion=True)
        from .checkpoint import CheckpointFork
        self.checkpoint_fork=CheckpointFork(graph,config)
        from .writeback import candidates
        self.writeback_casts=candidates(graph) if config.get('fuse_vector_cast_writeback',False) else {}
        from .vector_fusion import integer_mask_candidates
        self.integer_mask_fusions=(integer_mask_candidates(graph)
            if config.get('fuse_integer_mask_materialization',False) else {})
        from .vector_fusion import training_batch_norm_relu_candidates
        normalization_fusion_rows=config.get('training_batch_norm_fusion_min_rows',0)
        if (not isinstance(normalization_fusion_rows,int) or isinstance(normalization_fusion_rows,bool)
            or normalization_fusion_rows<0):
            raise ValueError('Training BatchNorm fusion row threshold must be nonnegative')
        self.training_batch_norm_relu_fusions=({at:target for at,target in
            training_batch_norm_relu_candidates(graph).items()
            if rows(graph.events[at]['outputs'][0])>=normalization_fusion_rows}
            if (coverage or demand) and config.get('fuse_training_batch_norm_relu',False) else {})
        from .vector_fusion import training_batch_norm_add_relu_candidates
        self.training_batch_norm_add_relu_fusions=({at:chain for at,chain in
            training_batch_norm_add_relu_candidates(graph).items()
            if rows(graph.events[at]['outputs'][0])>=normalization_fusion_rows}
            if (coverage or demand) and config.get('fuse_training_batch_norm_add_relu',False) else {})
        if self.checkpoint_fork.storage_bytes:
            config=dict(config,control_sram_bytes=config['control_sram_bytes']-self.checkpoint_fork.storage_bytes)
        if demand and config.get('demand_consumer_batches',False):
            # The original A controller compacts surviving output requests.
            # Execute those actual rows under one finite weight stripe,
            # using the existing feeder/row-chain state, without B ordering.
            config=dict(config,dense_outer_rows=True,
                        dense_output_columns=config['arrays']*config['array_cols'],
                        dense_columns_oversized_only=True,column_window_overlap=True,
                        dense_outer_row_chain=True,row_chain_weight_stripes=True,
                        operand_group_k_outer=True)
        if coverage and config.get('coverage_consumer_windows',False):
            # Complete local consumers of this coverage batch before moving
            # the feature-residency frontier. Reuse the existing finite row
            # chain; the declared matrix count and real cache spills remain.
            matrix_limit=(config.get('joint_row_chain_max_matrices') if demand and
                          'joint_row_chain_max_matrices' in config else
                          config.get('coverage_row_chain_max_matrices',config.get('row_chain_max_matrices',0)))
            if ('coverage_row_chain_max_matrices' in config and
                (not isinstance(matrix_limit,int) or isinstance(matrix_limit,bool) or matrix_limit<1)):
                raise ValueError('Coverage row chain matrix count must be a positive integer')
            config=dict(config,dense_outer_row_chain=True,
                        row_chain_weight_stripes=True,row_chain_max_matrices=matrix_limit)
        if coverage and config.get('coverage_private_consumer_bypass',False):
            if not config.get('coverage_private_bypass',False):
                raise ValueError('Private consumer bypass requires proven private coverage rows')
            # A row-private operator has no cross-consumer feature to retain.
            # Keep its ordinary finite feeder program as well as its native
            # row order; B remains active for operators with shared inputs.
            config=dict(config,coverage_private_consumer_bypass_active=True)
        if coverage and config.get('coverage_operand_batches',False):
            # Reuse each admitted K stripe across the bounded coverage batch.
            # The ordinary native schedule remains unchanged without B.
            config=dict(config,operand_group_k_outer=True)
        if coverage and config.get('coverage_operand_prefetch',False):
            if not config.get('coverage_operand_batches',False):
                raise ValueError('Coverage operand prefetch requires operand batches')
            config=dict(config,coverage_operand_prefetch_active=True)
        if (coverage or demand) and config.get('consumer_operand_prefetch',False):
            if not config.get('operand_group_k_outer',False):
                raise ValueError('Consumer operand prefetch requires admitted operand batches')
            config=dict(config,coverage_operand_prefetch_active=True)
        if coverage and config.get('coverage_retain_dense_accumulator',False):
            # Keep the current consumer's K-reduction on its existing PE
            # owner. The feeder's original retained path pins that array
            # until the true final partial commit; no extra partial copy.
            config=dict(config,retain_dense_accumulator=True,dense_operand_k_outer=True)
        if coverage and config.get('coverage_blocked_reduction',False):
            if not config.get('coverage_operand_batches',False):
                raise ValueError('Blocked coverage reductions require operand batches')
            config=dict(config,dense_blocked_reduction=True)
        if (coverage or demand) and config.get('consumer_blocked_reduction',False):
            if not config.get('operand_group_k_outer',False):
                raise ValueError('Blocked consumer reductions require admitted operand batches')
            config=dict(config,dense_blocked_reduction=True)
        if (coverage or demand) and 'consumer_k_tile' in config:
            depth=(config.get('joint_consumer_k_tile',config['consumer_k_tile'])
                   if coverage and demand else
                   config.get('full_depth_coverage_k_tile',config['consumer_k_tile'])
                   if coverage else config['consumer_k_tile'])
            if (not isinstance(depth,int) or isinstance(depth,bool) or depth<1 or
                    not config.get('operand_group_k_outer',False) or
                    not config.get('finite_operand_feeder',False)):
                raise ValueError('Consumer operand depth requires positive integer depth and finite admitted batches')
            # Invocation-static repartition of the existing output SRAM.
            # All clients, including diagnostic children, inherit these same
            # physical half-slots; accumulator capacity shrinks accordingly.
            config=dict(config,k_tile=depth)
        if (coverage or demand) and config.get('consumer_resident_route',False):
            if not config.get('operand_group_k_outer',False):
                raise ValueError('Resident consumer routing requires admitted operand batches')
            config=dict(config,direct_operand_stream=True,direct_translated_tiles=True,
                        direct_multicast=True,stream_operand_slices=True)
        if coverage and config.get('coverage_weight_batches',False):
            # B admits coverage-ready groups under a common weight stripe.
            # Reuse the existing finite N-outer executor; no extra buffer,
            # array, bank or weight payload is created by this controller.
            config=dict(config,dense_outer_rows=True,
                        dense_output_columns=config['arrays']*config['array_cols'],
                        dense_columns_oversized_only=True,column_window_overlap=True)
        if (coverage or demand) and config.get('consumer_all_weight_columns',False):
            if not (config.get('dense_outer_rows',False) and
                    config.get('operand_group_k_outer',False) and
                    config.get('finite_operand_feeder',False)):
                raise ValueError('Resident-weight columns require admitted finite consumer batches')
            # A resident weight matrix can still have wide output rows.
            # Reuse the original finite column program instead of forcing
            # small full-width M owners. No extra port, slot or accumulator.
            config=dict(config,dense_columns_oversized_only=False)
        if (coverage or demand) and config.get('consumer_sparse_weight_stripes',False):
            if not (config.get('finite_operand_feeder',False) and
                    config.get('consumer_copy_neighborhoods',False)):
                raise ValueError('Consumer sparse stripes require admitted finite neighborhood supply')
            # The sparse column executor further clips each stripe using the
            # actual operator weight bytes and the fixed weight SRAM capacity.
            config=dict(config,
                        sparse_output_columns=config['weight_sram_bytes']//SECTOR_BYTES)
        if (coverage or demand) and config.get('consumer_serpentine_columns',False):
            if not config.get('dense_outer_rows',False):
                raise ValueError('Serpentine columns require admitted dense column batches')
            config=dict(config,dense_serpentine_columns=True)
        if (coverage or demand) and 'consumer_stripe_arrays' in config:
            stripes=config['consumer_stripe_arrays']
            if (not isinstance(stripes,int) or isinstance(stripes,bool) or
                not 1<=stripes<=config['arrays'] or not config.get('dense_outer_rows',False)):
                raise ValueError('Consumer stripes require 1..arrays admitted column tiles')
            # Restrict a shared-consumer cohort's N domain; M tiles still
            # issue independently across every original array. The existing
            # output budget derives the corresponding finite M window.
            config=dict(config,dense_output_columns=stripes*config['array_cols'])
            if config.get('consumer_slot_stripes',False):
                raise ValueError('Consumer stripe width candidates are mutually exclusive')
        if (coverage or demand) and config.get('consumer_resident_rows',False):
            if not config.get('dense_outer_rows',False):
                raise ValueError('Consumer resident rows require admitted column batches')
            config=dict(config,consumer_resident_rows_active=True)
        if (coverage or demand) and config.get('consumer_slot_stripes',False):
            if not (config.get('dense_outer_rows',False) and config.get('operand_group_k_outer',False)):
                raise ValueError('Consumer slot stripes require admitted operand batches')
            # One current K weight per existing B half-slot; arrays may
            # consume them in several waves without adding a PE owner.
            config=dict(config,dense_output_columns=config['arrays']*
                        config.get('operand_buffers_per_array',2)*config['array_cols'])
        if (coverage or demand) and config.get('consumer_weight_retention',False):
            if not config.get('operand_group_k_outer',False):
                raise ValueError('Consumer weight retention requires admitted operand batches')
            config=dict(config,consumer_weight_retention_active=True)
        if (coverage or demand) and config.get('consumer_packed_operands',False):
            if not config.get('operand_group_k_outer',False):
                raise ValueError('Consumer operand packing requires admitted operand batches')
            config=dict(config,consumer_packed_operands_active=True)
        if (coverage or demand) and config.get('consumer_copy_interleave',False):
            if not (config.get('bounded_operand_copy',False) and config.get('operand_group_k_outer',False)):
                raise ValueError('Consumer copy interleave requires bounded admitted operand batches')
            config=dict(config,consumer_copy_batch_active=config['queue_entries'])
        if (coverage or demand) and config.get('consumer_dual_supply',False):
            if not all(config.get(flag,False) for flag in (
                    'consumer_copy_interleave','consumer_copy_pipeline','bounded_operand_copy',
                    'reuse_operand_tiles','independent_sram_ports')):
                raise ValueError('Dual operand supply requires admitted bounded independent sources')
            config=dict(config,consumer_dual_supply_active=True)
        if (coverage or demand) and config.get('consumer_feature_rhs_supply',False):
            if not (config.get('consumer_dual_supply_active',False) and
                    config.get('consumer_copy_register_control',False)):
                raise ValueError('Feature RHS supply requires the existing registered dual operand program')
            config=dict(config,consumer_feature_rhs_supply_active=True)
        if (coverage or demand) and config.get('consumer_mm_batches',False):
            if not all(config.get(flag,False) for flag in (
                    'operand_group_k_outer','bounded_operand_copy','reuse_operand_tiles')):
                raise ValueError('Consumer mm batches require admitted reusable bounded operands')
            config=dict(config,consumer_mm_batches_active=True)
        if (coverage or demand) and config.get('consumer_adjacent_supply',False):
            if not config.get('consumer_dual_supply_active',False):
                raise ValueError('Adjacent operand supply requires admitted dual operand supply')
            config=dict(config,consumer_adjacent_supply_active=True)
        if (coverage or demand) and config.get('consumer_copy_mask_pipeline',False):
            if not all(config.get(flag,False) for flag in (
                    'consumer_copy_interleave','consumer_copy_pipeline','consumer_copy_register_control')):
                raise ValueError('Mask pipeline requires admitted registered copy descriptors')
            if any(config.get(flag,False) for flag in (
                    'consumer_copy_ready_first','consumer_copy_source_credit','consumer_copy_source_defer')):
                raise ValueError('Mask pipeline requires the original sector-order controller')
            config=dict(config,consumer_copy_mask_pipeline_active=True)
        if (coverage or demand) and config.get('consumer_column_gelu',False):
            if not config.get('dense_outer_rows',False):
                raise ValueError('Consumer GELU requires admitted outer row windows')
            config=dict(config,dense_column_gelu=True)
        if (coverage or demand) and config.get('consumer_zero_seed',False):
            if not config.get('operand_group_k_outer',False):
                raise ValueError('Consumer zero seed requires admitted operand batches')
            config=dict(config,consumer_zero_seed_active=True)
        if (coverage or demand) and config.get('consumer_operand_skew',False):
            if not all(config.get(flag,False) for flag in (
                    'operand_group_k_outer','bounded_operand_copy','reuse_operand_tiles')):
                raise ValueError('Operand skew requires bounded admitted reusable operands')
            # Reuse the streamed operand bank permutation, only within whole
            # physical bank rows of the existing nonstream half-slots.
            bank_row=config['sram_banks']*SECTOR_BYTES
            spans=[lanes*config['k_tile']*4 for lanes in (config['array_rows'],config['array_cols'])]
            active=(not config.get('stream_operand_slices',False) and
                    not config.get('direct_operand_stream',False) and
                    all(span%bank_row==0 for span in spans))
            config=dict(config,consumer_operand_skew_active=active)
        if (coverage or demand) and config.get('consumer_attention_batches',False):
            if not (config.get('finite_operand_feeder',False) and
                    config.get('independent_sram_ports',False)):
                raise ValueError('Attention consumer batches require finite independent supply')
            # A/B's ready consumer cohort shares a finite K/V staging owner.
            # The existing output budget clips query rows before allocation;
            # every M tile still reads its K/V operands through real banks.
            config=dict(config,consumer_attention_batches_active=True,
                        attention_query_tiles=config['arrays'],
                        attention_register_softmax=True)
        if (coverage or demand) and config.get('consumer_attention_pipeline',False):
            if not config.get('consumer_attention_batches_active',False):
                raise ValueError('Attention query pipeline requires admitted consumer batches')
            config=dict(config,consumer_attention_pipeline_active=True)
        if (coverage or demand) and config.get('consumer_attention_patch_wavefront',False):
            if not config.get('consumer_attention_pipeline_active',False):
                raise ValueError('Attention patch wavefront requires the admitted query pipeline')
            config=dict(config,attention_patch_wavefront=True)
        if (coverage or demand) and config.get('consumer_attention_qk_frontload',False):
            if not config.get('consumer_attention_pipeline_active',False):
                raise ValueError('Frontloaded QK requires the admitted query pipeline')
            config=dict(config,consumer_attention_qk_frontload_active=True)
        if (coverage or demand) and config.get('consumer_attention_matrix_queue',False):
            if not config.get('consumer_attention_pipeline_active',False):
                raise ValueError('Consumer matrix queues require the admitted query pipeline')
            config=dict(config,attention_ready_queue=True)
        if (coverage or demand) and config.get('consumer_attention_reclaim',False):
            if (not config.get('consumer_attention_pipeline_active',False) or
                    config.get('stream_operand_slices',False) or
                    config.get('direct_operand_stream',False)):
                raise ValueError('Attention storage reuse requires the admitted nonstream query pipeline')
            config=dict(config,consumer_attention_reclaim_active=True)
        if (coverage or demand) and config.get('consumer_attention_queue_batches',False):
            if not config.get('consumer_attention_pipeline_active',False):
                raise ValueError('Queued Attention batches require the admitted query pipeline')
            # The original task queue bounds query descriptors; actual output
            # storage still clips each head/patch cohort before allocation.
            # This changes no arrays, ports or K/V traffic per query tile.
            config=dict(config,attention_query_tiles=config['queue_entries'])
        if coverage and config.get('coverage_attention_staging',False):
            if not (config.get('coverage_attention_links',False) and
                    config.get('consumer_attention_pipeline_active',False)):
                raise ValueError('Coverage staging requires witnessed Attention links and query pipeline')
        if coverage and config.get('coverage_attention_fixed_layout',False) and not config.get('coverage_attention_staging',False):
            raise ValueError('Fixed Attention layout requires witnessed K/V staging')
        if config.get('consumer_copy_register_control',False) and not config.get('consumer_copy_interleave',False):
            raise ValueError('Register copy control requires an admitted descriptor batch')
        if config.get('consumer_copy_pipeline',False) and not config.get('consumer_copy_interleave',False):
            raise ValueError('Pipelined copy control requires an admitted descriptor batch')
        if config.get('consumer_copy_ready_first',False) and not config.get('consumer_copy_interleave',False):
            raise ValueError('Ready-first copy control requires an admitted descriptor batch')
        if config.get('consumer_copy_source_credit',False):
            if not (config.get('consumer_copy_register_control',False) and config.get('consumer_copy_pipeline',False)):
                raise ValueError('Source-credit copy control requires pipelined register descriptors')
            if config.get('consumer_copy_ready_first',False):
                raise ValueError('Source-credit and sector ready-first are alternative schedulers')
        if config.get('consumer_copy_source_defer',False):
            if not (config.get('consumer_copy_register_control',False) and config.get('consumer_copy_pipeline',False)):
                raise ValueError('Deferred source requires pipelined register descriptors')
            if config.get('consumer_copy_source_credit',False) or config.get('consumer_copy_ready_first',False):
                raise ValueError('Deferred source requires the descriptor round-robin scheduler')
        if (coverage or demand) and 'consumer_reduction_slices' in config:
            slices=config['consumer_reduction_slices']
            if (not isinstance(slices,int) or isinstance(slices,bool) or slices<1 or
                    config['k_tile']%slices or
                    (slices!=1 and not config.get('operand_group_k_outer',False))):
                raise ValueError('Consumer reduction slices must divide an admitted operand tile')
            if slices!=1 and (config.get('stream_operand_slices',False) or config.get('consumer_packed_operands',False)):
                raise ValueError('Consumer reduction slicing requires ordinary unpacked operands')
            config=dict(config,consumer_reduction_slices_active=slices)
        if config.get('diagnostic_history_fp16',False) and not (
                config.get('diagnostic_direct_state',False) and
                config.get('finite_diagnostic_pipeline',False) and
                config.get('trajectory_space')=='query_scores'):
            raise ValueError('Packed query history requires finite direct query state')
        if config.get('diagnostic_terminal_state',False) and not (
                config.get('diagnostic_direct_state',False) and
                config.get('finite_diagnostic_pipeline',False) and
                config.get('trajectory_space')=='query_scores'):
            raise ValueError('Terminal trajectory state requires finite direct query state')
        if config.get('direct_translated_tiles',False) and not config.get('direct_operand_stream',False):
            raise ValueError('Translated tiles require direct operand streaming')
        if config.get('direct_operand_stream',False) and not config.get('stream_operand_slices',False):
            raise ValueError('Direct operands require budgeted sector input latches')
        if config.get('direct_operand_stream',False):
            distance=config.get('direct_prefetch_bytes',0)
            if not isinstance(distance,int) or isinstance(distance,bool) or distance<0 or distance%DMA_BURST_BYTES:
                raise ValueError('Direct prefetch distance must be a nonnegative number of complete DMA bursts')
            if config.get('direct_multicast',False) and config['arrays']>64:
                raise ValueError('Direct receiver mask supports at most 64 arrays')
        if config.get('dense_outer_rows',False) and not config.get('finite_operand_feeder',False):
            raise ValueError('Dense outer rows require the finite operand feeder')
        if config.get('diagnostic_window_overlap',False) and not config.get('diagnostic_row_windows',False):
            raise ValueError('Diagnostic window overlap requires finite row windows')
        if config.get('diagnostic_source_fusion',False) and not config.get('bounded_operand_copy',False):
            raise ValueError('Diagnostic source fusion requires bounded bank-to-bank copy')
        if config.get('diagnostic_source_direct',False) and not config.get('diagnostic_source_fusion',False):
            raise ValueError('Direct diagnostic sources require finite source fusion')
        if config.get('stream_operand_slices',False) and not all(config.get(flag,False)
                for flag in ('finite_operand_feeder','reuse_operand_tiles','array_wave_pipeline')):
            raise ValueError('K streaming requires finite reusable operands and the existing array wave pipeline')
        if config.get('interleave_operand_slices',False) and not config.get('stream_operand_slices',False):
            raise ValueError('Interleaved microprograms require streamed operand slices')
        if config.get('bounded_operand_copy',False) and not all(config.get(flag,False)
                for flag in ('finite_operand_feeder','reuse_operand_tiles')):
            raise ValueError('Bounded operand copy requires finite reusable operand buffers')
        copy_groups=config.get('copy_register_groups',1)
        if not isinstance(copy_groups,int) or copy_groups<1:
            raise ValueError('Copy register groups must be a positive integer')
        query_tiles=config.get('attention_query_tiles',1)
        if not isinstance(query_tiles,int) or isinstance(query_tiles,bool) or query_tiles<1:
            raise ValueError('Attention query tiles must be a positive integer')
        prefetch=config.get('prefetch_microprogram_tiles',1)
        if not isinstance(prefetch,int) or not 1<=prefetch<=config.get('operand_buffers_per_array',2):
            raise ValueError('Microprogram prefetch must fit existing operand buffers')
        if config.get('input_latch_budget','output') not in ('output','control'):
            raise ValueError('Input latch budget must be output or control')
        if config.get('elide_row_concat',False) or config.get('elide_channel_concat',False):
            from .concatenation import lower
            graph=lower(graph,channels=config.get('elide_channel_concat',False),
                        row_chunks=config.get('elide_row_concat',False))
        self.g=graph;self.c=config;self.a=demand;self.b=coverage;self.order=order
        self.coverage_row_chain_limit=config.get('row_chain_max_matrices')
        self.pilot_row_chain_limit=(pilot_row_chain_limit if coverage and demand and
            config.get('coverage_bypass_pilot',False) else self.coverage_row_chain_limit)
        self.coverage_phase_active=coverage
        # One directed candidate-pair bitmap in the existing control SRAM.
        # Python's transient set below is its host representation, not free
        # unbounded queue storage.
        self.coverage_link_state_bytes=(math.ceil(config['queue_entries']**2/8)
            if coverage and config.get('coverage_linked_supply',False) else 0)
        self.memory=Memory(config,coverage)
        self.matrix=np.zeros(config['arrays'],dtype=np.int64)
        self.vector=0;self.cycle=0;self.stats=Counter();self.stages=Counter()
        if self.coverage_link_state_bytes:
            self.stats['coverage_link_bitmap_bytes']=self.coverage_link_state_bytes
        from .ready_queue import ReadyQueue
        self.vector_queue=ReadyQueue(config['queue_entries'])
        self.vector_queue_storage_bytes=(config['queue_entries']*ReadyQueue.entry_bytes
            if config.get('vector_ready_queue',False) else 0)
        self.attention_queues=None
        attention_entries=config.get('attention_program_entries',config['queue_entries'])
        if (not isinstance(attention_entries,int) or isinstance(attention_entries,bool) or
            attention_entries<1):
            raise ValueError('Attention programs require a positive finite integer entry count')
        self.attention_queue_capacity=attention_entries//config['arrays']
        if 'attention_program_entries' in config and self.attention_queue_capacity<1:
            raise ValueError('Explicit Attention programs require at least one entry per array')
        original_capacity=max(1,config['queue_entries']//config['arrays'])
        # Preserve the incumbent selector. Extra tree levels register their
        # candidate before admission, with one ordinary descriptor per stage.
        self.attention_queue_extra_select_stages=max(0,
            (max(1,self.attention_queue_capacity)-1).bit_length()-(original_capacity-1).bit_length())
        self.attention_queue_storage_bytes=((self.attention_queue_capacity+self.attention_queue_extra_select_stages)*
            config['arrays']*ReadyQueue.entry_bytes if config.get('attention_ready_queue',False) else 0)
        from .attention import register_bytes
        self.softmax_register_storage_bytes=register_bytes(config)
        self.softmax_ready=np.zeros(2,dtype=np.int64)
        query_descriptors=(config['attention_query_tiles']
            if config.get('consumer_attention_pipeline_active',False) else config['arrays'])
        self.attention_pipeline_storage_bytes=(query_descriptors*ReadyQueue.entry_bytes
            if config.get('consumer_attention_pipeline_active',False) else 0)
        self.attention_pipeline_ready=np.zeros(query_descriptors,dtype=np.int64)
        from .attention import staging_bytes
        self.attention_staging_storage_bytes=(staging_bytes(config)
            if coverage and config.get('coverage_attention_staging',False) else 0)
        self.attention_staging_leases=[None]*config['output_buffers']
        self.attention_staging_permission=0
        self.attention_output_layout_active=False
        if self.attention_staging_storage_bytes:
            self.stats['attention_coverage_staging_descriptor_bytes']=self.attention_staging_storage_bytes
        from .diagnostic_pipeline import source_register_words
        self.diagnostic_source_register_words=source_register_words(graph,config)
        # Two FP32 data/owner contexts, three scalar reduction words and the
        # existing 24-byte ready descriptor per context. Paid from control SRAM.
        self.diagnostic_source_register_storage_bytes=(2*(self.diagnostic_source_register_words*4+12+ReadyQueue.entry_bytes)
            if self.diagnostic_source_register_words else 0)
        self.diagnostic_source_ready=np.zeros(2,dtype=np.int64)
        if config.get('attention_ready_queue',False) and self.attention_queue_capacity<1:
            raise ValueError('Attention ready queue needs at least one entry per array')
        from .feeder import stream_descriptor_bytes,input_latch_bytes
        self.microprogram_storage_bytes=stream_descriptor_bytes(config)
        from .direct import descriptor_bytes,translation_bytes
        self.direct_descriptor_storage_bytes=descriptor_bytes(config)
        self.direct_translation_storage_bytes=translation_bytes(config)
        from .copyroute import register_bytes,descriptor_register_bits
        self.copy_register_storage_bytes=register_bytes(config)
        self.copy_descriptor_storage_bytes=config.get('consumer_copy_batch_active',0)*32
        descriptor_count=config.get('consumer_copy_batch_active',0)
        self.copy_descriptor_queue=(np.zeros(descriptor_count+1,np.int64)
            if descriptor_count and config.get('consumer_copy_pipeline',False) else None)
        if self.copy_descriptor_queue is not None:
            link_bits=config['sram_banks']*copy_groups*max(1,(descriptor_count-1).bit_length())
            link_bytes=math.ceil(link_bits/(8*SECTOR_BYTES))*SECTOR_BYTES
            self.copy_descriptor_storage_bytes+=SECTOR_BYTES+link_bytes
        if config.get('consumer_dual_supply_active',False):
            # Two 64-bit tag cursors; tile address generators and the shared
            # queue already carry both operand identities and source roles.
            self.copy_descriptor_storage_bytes+=16
        if descriptor_count and config.get('consumer_copy_ready_first',False):
            # One pending bit per descriptor plus a 64-bit sector/controller
            # word. No extra source payload or physical bank queue is added.
            self.copy_descriptor_storage_bytes+=math.ceil((math.ceil(descriptor_count/8)+8)/SECTOR_BYTES)*SECTOR_BYTES
        self.copy_descriptor_register_bits=descriptor_register_bits(config)
        if self.copy_descriptor_register_bits>8*self.copy_descriptor_storage_bytes:
            raise ValueError('Copy descriptor register fields exceed their reserved bytes')
        self.copy_ready=np.zeros((config['sram_banks'],copy_groups),np.int64)
        self.copy_source_order=(np.zeros(config['sram_banks'],np.int64)
            if descriptor_count and config.get('consumer_copy_source_defer',False) else None)
        if self.copy_source_order is not None:
            # One finite source-issuer cursor/control word per shared route
            # bank. Host completion timestamps stand for actual read events.
            self.copy_descriptor_storage_bytes+=self.copy_source_order.nbytes
            # A copy source FIFO entry identifies its existing descriptor.
            # Both feature and weight bank groups need real return identity;
            # command addresses/ready fields belong to the original queues.
            source_groups=2 if config.get('independent_sram_ports',False) else 1
            fifo_bits=source_groups*config['sram_banks']*config['queue_entries']*max(1,(descriptor_count-1).bit_length())
            self.copy_descriptor_storage_bytes+=math.ceil(fifo_bits/(8*SECTOR_BYTES))*SECTOR_BYTES
        self.input_latch_control_bytes=(input_latch_bytes(config)
            if config.get('input_latch_budget','output')=='control' else 0)
        if self.control_capacity()<=0:raise ValueError('Vector issue queue exhausts control SRAM')
        self.tile_orders={}
        self.coverage_rows=None
        if coverage and order is not None and config.get('coverage_exact_rows',False):
            from .coverage_rows import CoverageRows
            self.coverage_rows=CoverageRows(graph,order,config['coverage_group_rows'],
                coverage_links,config.get('coverage_link_limit'),
                attention_links=config.get('coverage_attention_links',False),
                layout_domain_reuse=config.get('coverage_layout_domains',False),
                link_domain_reuse=config.get('coverage_link_domains',False))
        self.buffer_ready=np.zeros(config['output_buffers'],dtype=np.int64)
        self.feeder_ready=np.zeros(config['arrays']*config.get('operand_buffers_per_array',2),dtype=np.int64)
        self.feeder_b_ready=np.zeros_like(self.feeder_ready)
        self.input_latches=np.zeros((config['arrays'],2),dtype=np.int64)
        self.prepared_maps=set()
        self.prepared_readouts=set()
        self.last_use={}
        self.produced_versions={key(s) for e in graph.events for s in e['outputs']}
        for i,event in enumerate(graph.events):
            # indexed_store overwrites its first operand at explicit row IDs;
            # inputs_for carries actual old-byte inheritance separately.
            for spec in event['inputs'][int(event['kind']=='indexed_store'):]:self.last_use[key(spec)]=i
        self.protected=set()
        self.protected.add(key(graph.events[-1]['outputs'][0]))
        self.phase_suppliers=None
        self.event_profile=[]
        # Host representation of the current decision's two bounded producer
        # counters. Other events require no persistent completion table.
        self.diagnostic_producers={at for barrier in graph.maps.get('diagnostic_control_barriers',{}).values()
                                   for at in barrier.get('producers',[])}
        self.diagnostic_completions={}
        self.push_plans={}
        self.chain_floor=None
        self.chain_end=True
        self.active_output_buffer=0
        self.diagnostic_cast_ready=0
        self.coverage_preparation_ready=None
        self.coverage_preparation_meta=None
        self.coverage_preparation_start=0
        self.coverage_preparation_total_vector=0
        self.coverage_preparation_scheduled_vector=0
        self.coverage_preparation_vector_ready=0
        self.coverage_preparation_products=[]
        self.coverage_preparation_available_bytes=0
        self.coverage_preparation_available_ready=0
        self.coverage_preparation_memory_ready=0
        self.coverage_preparation_written_bytes=0
        from .epilogue import plans as epilogue_plans
        self.column_epilogues=epilogue_plans(graph,config)
        from .activity import Activity
        self.activity=Activity(config['arrays']) if config.get('observe_execution_intervals',False) else None

    def issue_vector(self,earliest,duration):
        """Return this program's completion, separately from the unit horizon."""
        earliest=max(earliest,self.diagnostic_cast_ready)
        if not self.c.get('vector_ready_queue',False):
            done=max(self.vector,earliest)+duration
        else:
            done=self.vector_queue.reserve(earliest,duration,max(self.cycle,self.memory.clock))
            # Full-queue admission stalls the shared dispatch, not only the
            # new program. Older reservations continue to own their intervals.
            self.memory.clock=max(self.memory.clock,self.vector_queue.floor)
        self.vector=max(self.vector,done)
        if self.activity is not None:self.activity.add_vector(done-duration,done)
        return done

    def issue_matrix(self,count,duration,earliest,initiation=0):
        activity=self.activity.matrix_buffer(count) if self.activity is not None else None
        if self.attention_queues is None:
            done=int(self.memory.core.matrix_tiles(self.matrix,count,duration,earliest,initiation,activity))
        else:
            occupied=initiation or duration;done=earliest
            admission=self.c['issue_cycles']+self.attention_queue_extra_select_stages
            for tile in range(count):
                issued=max(self.cycle,self.memory.clock)
                starts=[q.preview(earliest,occupied,issued,admission) for q in self.attention_queues]
                unit=int(np.argmin(starts));queue=self.attention_queues[unit]
                end=queue.reserve(earliest,occupied,issued,admission)
                self.memory.clock=max(self.memory.clock,queue.floor)
                self.matrix[unit]=max(self.matrix[unit],end)
                done=max(done,end+duration-occupied)
                if activity is not None:activity[tile]=[end-occupied,end,unit]
                self.stats['attention_queue_peak']=max(self.stats['attention_queue_peak'],sum(
                    sum(end>self.memory.clock for _,end in q.pending) for q in self.attention_queues))
                self.stats['attention_queue_admission_cycles']+=admission
        if activity is not None:self.activity.add_matrix(activity,done)
        return done

    def control_capacity(self):
        return (self.c['control_sram_bytes']-self.vector_queue_storage_bytes
                -self.coverage_link_state_bytes
                -self.attention_staging_storage_bytes
                -self.copy_descriptor_storage_bytes
            -self.attention_queue_storage_bytes-self.softmax_register_storage_bytes
            -self.diagnostic_source_register_storage_bytes-self.attention_pipeline_storage_bytes
            -self.microprogram_storage_bytes-self.input_latch_control_bytes
            -self.direct_descriptor_storage_bytes
            -self.direct_translation_storage_bytes
            -self.copy_register_storage_bytes-getattr(self,'normalization_stream_bytes',0))

    def accumulator_base(self,element_bytes):
        """A physical output owner cannot alias the finite operand slots."""
        if self.attention_output_layout_active:return self.active_output_buffer*self.output_budget()
        if not self.c.get('finite_operand_feeder',False):return self.active_output_buffer*self.output_budget()
        from .feeder import operand_capacity,input_latch_bytes
        slots,capacity,budget=operand_capacity(self.c,element_bytes)
        latch_charge=input_latch_bytes(self.c) if self.c.get('input_latch_budget','output')=='output' else 0
        return slots*capacity+latch_charge+self.active_output_buffer*budget

    def allocate_workspace(self,size):
        """One physical address space for invocation-local control assets."""
        if not hasattr(self,'workspace_end'):
            end=max((base+self.g.storage[k[0]]['bytes'] for k,base in self.g.bases.items()),default=0)
            for name,base in self.g.map_bases.items():
                rule=self.g.maps['maps'][name]
                size_=np.asarray(rule['mapping']).nbytes if 'mapping' in rule else sum(a.nbytes+b.nbytes for a,b in rule.get('offsets',[]))
                end=max(end,base+size_)
            self.workspace_end=math.ceil(end/SLOT_BYTES)*SLOT_BYTES
        address=self.workspace_end;self.workspace_end+=math.ceil(size/SLOT_BYTES)*SLOT_BYTES
        return address

    def final_gather_plan(self,index,producer):
        """Invert only a real final row selector, with bounded build scratch."""
        if not self.c.get('stream_final_gather',False):return None
        e=self.g.events[index];root=self.g.events[-1]['outputs'][0]
        indices=self.g.maps.get('indices',{}).get(index,[])
        if (e.get('op')!='aten.index.Tensor' or len(e['outputs'])!=1 or len(indices)!=1
            or key(e['outputs'][0])!=key(root) or len(root['shape'])!=2):return None
        if not e.get('row_selector',False) and not any(s.get('host_index',False) for s in e['inputs'][1:]):return None
        source=e['inputs'][0]
        start=self.g.bases[key(source)]+source['address']-self.g.storage[source['storage']]['address']
        if (self.last_use.get(key(source))!=index or not source['contiguous']
            or start%SECTOR_BYTES
            or any(source[f]!=producer[f] for f in ('storage','version','address','shape','stride','element_bytes'))
            or len(source['shape'])!=2 or source['shape'][-1]!=root['shape'][-1]
            or source['shape'][-1]*source['element_bytes']%SECTOR_BYTES):return None
        mapping=np.asarray(indices[0]).reshape(-1).astype(np.int64)
        n=rows(source)
        if len(mapping)!=rows(root) or np.any(mapping>=n) or np.any(mapping < -n):return None
        raw_mapping=mapping
        mapping=mapping%n
        consumers=None
        if self.c.get('compact_dispatch_gather',False) and 'dispatch_ids' in e:
            # These are the current microprogram's explicit requests, already
            # produced by its decision. Never infer them from future closure.
            consumers=np.asarray(e['dispatch_ids'],dtype=np.int64)
            if (consumers.ndim!=1 or len(np.unique(consumers))!=len(consumers)
                or np.any(consumers<0) or np.any(consumers>=rows(root))):
                raise ValueError('Invalid explicit gather consumers')
            mapping=mapping[consumers]
            raw_mapping=raw_mapping[consumers]
        exact_dispatch=bool(e.get('dispatch_complete',False))
        if exact_dispatch and consumers is None:
            raise ValueError('Complete dispatch requires an explicit compact request list')
        from .feeder import inverse_row_scratch
        layout=inverse_row_scratch(len(mapping),n);scratch=layout[-1]
        if scratch>self.c['output_sram_bytes']//self.c['output_buffers'] or len(mapping)>=2**32:return None
        if index not in self.push_plans:
            order=np.argsort(mapping,kind='stable')
            if consumers is not None:order=consumers[order]
            offsets=np.r_[0,np.cumsum(np.bincount(mapping,minlength=n))]
            size=(len(mapping)+n+1)*4
            self.push_plans[index]=dict(order=order,offsets=offsets,source=source,
                base=self.allocate_workspace(size),bytes=size,scratch=scratch,
                layout=layout,mapping=mapping,raw_mapping=raw_mapping,consumer_ids=consumers,prepared=False,bitmap_reserved=False,
                exact_dispatch=exact_dispatch)
        return self.push_plans[index]

    def protected_storage(self,spec):
        """Aliases of a retained physical allocation have equal ownership."""
        base=self.allocation_base(key(spec))
        return any(self.allocation_base(identity)==base for identity in self.protected)

    def allocation_base(self,identity):
        owners=getattr(self.g,'concat_owners',{})
        return self.g.bases[owners.get(identity,identity)]

    def same_allocation(self,left,right):
        return self.allocation_base(key(left))==self.allocation_base(key(right))

    def private_phase_temporaries(self,carry=()):
        """Compile single-consumer row-local lifetimes, without future stops.

        The pilot and remaining consumer rows are disjoint. A private version
        produced for a pilot row therefore has no later-phase reader. Actual
        retirement still waits for that row's existing last-use/sector checks.
        Shared, aliased, partial and globally consumed versions stay protected.
        """
        producers={};users={};aliases={}
        for index,event in enumerate(self.g.events):
            for spec in event['outputs']:
                producers.setdefault(key(spec),[]).append((index,spec))
            for spec in event['inputs']:
                users.setdefault(key(spec),[]).append((index,spec))
            for spec in event['inputs']+event['outputs']:
                aliases.setdefault(self.allocation_base(key(spec)),set()).add(key(spec))
        excluded={key(self.g.events[-1]['outputs'][0]),*carry}
        excluded.update(key(spec) for cp in self.g.maps.get('checkpoints',[])
                        for spec in cp.get('readout_inputs',[])+([cp['spec']] if 'spec' in cp else []))
        result=set()
        for identity,definitions in producers.items():
            uses=users.get(identity,[])
            if identity in excluded or len(definitions)!=1 or len(uses)!=1:continue
            defined,source=definitions[0];used,operand=uses[0]
            producer=self.g.events[defined];consumer=self.g.events[used]
            if (defined>=used or len(producer['outputs'])!=1 or source['role']!='feature'
                or len(source['shape'])!=2 or not source['contiguous']
                or aliases[self.allocation_base(identity)]!={identity}):continue
            storage=self.g.storage[source['storage']]
            # Whole, identical views rule out inherited bytes and shifted or
            # lowered concat aliases. No run-time per-row ownership table.
            if (source['address']!=storage['address'] or source['bytes']!=storage['bytes']
                or any(source[field]!=operand[field] for field in
                       ('address','shape','stride','element_bytes','bytes','contiguous'))):continue
            # Only an explicitly injective ordinary consumer is admitted.
            # Linear RHS/bias and global BatchNorm statistics are never local,
            # even when their leading dimension happens to match output rows.
            local=(consumer['kind']=='linear' and operand is consumer['inputs'][0]
                   and not consumer.get('indexed_source',False)) or (
                consumer['kind']=='aten' and consumer.get('op','').startswith((
                    'aten.add.','aten.sub.','aten.mul.','aten.div.','aten.relu',
                    'aten.gelu','aten.layer_norm','aten.to.','aten.clamp_min',
                    'aten.linalg_vector_norm')))
            if (not local or not self.can_reclaim(consumer,operand,self.same_allocation)
                or 'diagnostic_source_rows' in consumer
                or any(self.same_allocation(out,operand) for out in consumer['outputs'])
                or producer['kind'] in ('indexed_store','progressive_update','progressive_observe')
                or any(self.same_allocation(inp,source) for inp in producer['inputs'])):continue
            result.add(identity)
        return result

    def allow_phase_retirement(self,control):
        return control is None or control[3]!='pilot' or self.phase_suppliers is not None

    def prepare_final_gather(self,index,plan):
        if plan['prepared']:return
        m=self.memory;c=self.c;e=self.g.events[index]
        before=self.cycle
        # This optional layout owns only the selected original accumulator.
        # Its scratch sits beyond all feeder slots. Other output/operand
        # owners therefore need not drain; their calendars remain live.
        local=(c.get('diagnostic_dispatch_owner',False) and plan.get('exact_dispatch',False)
               and plan['scratch']<=self.output_budget())
        owner=self.active_output_buffer;scratch_base=self.accumulator_base(4) if local else 0
        ready=max(self.cycle,m.clock,int(self.buffer_ready[owner])) if local else max(
            self.cycle,m.clock,int(self.buffer_ready.max()),int(self.feeder_ready.max()),int(self.feeder_b_ready.max()))
        for s in e['inputs'][1:]:
            if s['role']=='index':ready=max(ready,m.access(self.g.sectors(s,np.arange(rows(s)))))
        from .feeder import inverse_row_transfer,identity_row_transfer
        banks,calendar,lengths=m.ports(2)
        begun=max(ready,self.vector)
        fast=False;ready=begun;vector_cycles=0
        payload=[s for s in e['inputs'][1:] if s['role']=='index']
        paired=(len(payload)==1 and payload[0]['shape']==[len(plan['mapping']),2]
                and payload[0]['element_bytes']==4 and payload[0]['contiguous'])
        if (c.get('diagnostic_identity_dispatch',False) and plan.get('exact_dispatch',False)
                and paired and len(plan['mapping'])==len(plan['offsets'])-1):
            fast,ready,vector_cycles,offsets,constructed=identity_row_transfer(
                plan['raw_mapping'],len(plan['offsets'])-1,plan['layout'],ready,c['vector_lanes'],
                m.state,banks,calendar,lengths,c['bank_bytes_per_cycle'],m.counts,plan['consumer_ids'],scratch_base)
            self.stats['gather_identity_probe_rows']+=len(plan['mapping'])
            self.stats['gather_identity_programs']+=int(fast)
            self.stats['gather_identity_rows']+=len(plan['mapping']) if fast else 0
        if not fast:
            ready,work,offsets,constructed=inverse_row_transfer(plan['mapping'],len(plan['offsets'])-1,plan['layout'],
                ready,c['vector_lanes'],m.state,banks,calendar,lengths,c['bank_bytes_per_cycle'],m.counts,plan['consumer_ids'],scratch_base)
            vector_cycles+=work
        if not np.array_equal(offsets,plan['offsets']) or not np.array_equal(constructed,plan['order']):
            raise AssertionError('Timed inverse construction changed CSR consumers')
        ready=int(ready)
        # This compound vector/SRAM program conservatively keeps the unit
        # throughout its memory waits; no untracked late vector step remains.
        if self.c.get('vector_ready_queue',False):
            end=self.vector_queue.reserve(begun,ready-begun,begun)
            if end!=ready:raise AssertionError('Compound vector program missed its exclusive interval')
        self.vector=max(self.vector,ready);self.stats['vector_cycles']+=int(vector_cycles)
        if self.activity is not None:self.activity.add_vector(begun,ready)
        sectors=np.arange(plan['base']//SECTOR_BYTES,math.ceil((plan['base']+plan['bytes'])/SECTOR_BYTES))
        ready=m.access(sectors,write=True,ready=ready)
        self.cycle=max(self.cycle,ready);m.clock=max(m.clock,self.cycle)
        if local:
            self.buffer_ready[owner]=max(self.buffer_ready[owner],self.cycle)
            self.stats['gather_owner_local_programs']+=1
            self.stats['gather_owner_local_peak_bytes']=max(self.stats['gather_owner_local_peak_bytes'],plan['scratch'])
        else:self.buffer_ready[:]=np.maximum(self.buffer_ready,self.cycle)
        self.stats['gather_inversion_cycles']+=self.cycle-before
        self.stats['gather_inversion_bytes']+=plan['bytes']
        self.stats['gather_inversion_scratch_bytes']=max(self.stats['gather_inversion_scratch_bytes'],plan['scratch'])
        self.stats['gather_inversion_input_rows']+=len(plan['mapping'])
        plan['prepared']=True

    def initialize_gather_demand(self,selected,total):
        """Build each phase's bitmap from causal shared-state output requests."""
        ports=self.c['cam_lookups_per_cycle'];words=math.ceil(total/32)
        before=self.cycle;cost=math.ceil(words/ports)
        # Every request ID is read from the existing shared state. The
        # controller serializes conflicting words in each finite port batch;
        # a bit update takes read -> vector OR -> control-SRAM write.
        vector_cycles=0
        for begin in range(0,len(selected),ports):
            ids=selected[begin:begin+ports];banks=(ids//32)%ports
            rounds=int(np.bincount(banks,minlength=ports).max())
            cost+=1+2*rounds # ID load, control read and write
            arithmetic=rounds*math.ceil(min(ports,len(ids))/self.c['vector_lanes'])
            cost+=arithmetic;vector_cycles+=arithmetic
        self.cycle=self.issue_vector(max(self.cycle,self.memory.clock,self.vector),cost)
        self.memory.clock=self.cycle
        self.stats['vector_cycles']+=vector_cycles
        self.stats['gather_demand_setup_cycles']+=self.cycle-before
        self.stats['gather_demand_control_read_bytes']+=len(selected)*8
        self.stats['gather_demand_control_write_bytes']+=words*4+len(selected)*4

    def row_order(self,event_index,selected):
        if not self.coverage_phase_active or self.order is None:return selected
        if self.a and self.c.get('joint_compact_row_order',False):return selected
        e=self.g.events[event_index]
        if e.get('diagnostic_affine_scatter',False):return selected
        if self.private_coverage_rows(e):return selected
        n=rows(e['outputs'][0])
        if self.coverage_rows is not None:
            candidate=self.coverage_rows.layout(e['outputs'][0])
            self.stats['coverage_proven_views']=len(self.coverage_rows.orders)
        else:
            if n not in self.tile_orders:
                self.tile_orders[n]=coverage_layout(self.g.maps,self.order,n,self.c['coverage_group_rows'])
            candidate=self.tile_orders[n]
        if candidate is None:return selected
        active=np.zeros(n,bool);active[selected]=True
        return candidate[active[candidate]]

    def private_coverage_rows(self,event):
        if not self.coverage_phase_active or not self.c.get('coverage_private_bypass',False):return False
        from .coverage_rows import CoverageRows
        return CoverageRows.private_rows(event)

    def install_predictor(self,nbytes):
        """Cold-load frozen tables via shared finite DMA into control SRAM.

        The existing lookup lanes each accept one 32-bit word per cycle.
        A returned burst drains before its successor overwrites that control
        write path; the feature/weight SRAMs never acquire table ownership.
        """
        m=self.memory;c=self.c;start=self.cycle
        control_ready=self.cycle
        for begin in range(0,nbytes,DMA_BURST_BYTES):
            returned=m.core.transaction(self.cycle,False,m.state,m.pending,m.counts,
                m.burst_cycles,c['dram_latency_cycles'],DMA_BURST_BYTES,m.dma_activity)
            payload=min(DMA_BURST_BYTES,nbytes-begin)
            control_ready=max(control_ready,returned)+math.ceil(payload/(4*c['cam_lookups_per_cycle']))
        self.stats['predictor_table_bytes']=nbytes
        self.stats['control_sram_write_bytes']+=nbytes
        self.cycle=control_ready
        self.stats['predictor_setup_cycles']=self.cycle-start
        m.clock=max(m.clock,self.cycle)

    def install_coverage_preparation(self,metadata):
        """Install a bounded geometry-preparation program for fine-grained issue."""
        if not self.b:raise ValueError('Coverage preparation requires B')
        if self.coverage_preparation_meta is not None:
            if metadata!=self.coverage_preparation_meta:
                raise ValueError('Coverage preparation metadata changed')
            return
        scalar_ops=metadata.get('scalar_ops');nbytes=metadata.get('link_bytes')
        if (not isinstance(scalar_ops,int) or isinstance(scalar_ops,bool) or scalar_ops<0 or
            not isinstance(nbytes,int) or isinstance(nbytes,bool) or nbytes<0):
            raise ValueError('Invalid coverage preparation work')
        self.coverage_preparation_start=self.cycle
        self.coverage_preparation_total_vector=math.ceil(scalar_ops/self.c['vector_lanes'])
        self.coverage_preparation_meta=dict(metadata)
        self.stats['coverage_preparation_vector_cycles']=self.coverage_preparation_total_vector
        self.stats['coverage_preparation_metadata_bytes']=nbytes

    def service_coverage_preparation(self,force=False):
        """Issue one geometry microprogram and retire produced DMA packets."""
        if self.coverage_preparation_meta is None or self.coverage_preparation_ready is not None:return
        total=self.coverage_preparation_total_vector
        scheduled=self.coverage_preparation_scheduled_vector
        if scheduled<total:
            chunk=min(self.c.get('coverage_preparation_chunk_cycles',256),total-scheduled)
            if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<1:
                raise ValueError('Coverage preparation chunks must be positive integer cycles')
            done=self.issue_vector(self.cycle,chunk)
            self.coverage_preparation_vector_ready=max(
                self.coverage_preparation_vector_ready,done)
            scheduled+=chunk
            self.coverage_preparation_scheduled_vector=scheduled
            self.stats['vector_cycles']+=chunk
            nbytes=self.coverage_preparation_meta['link_bytes']
            produced=nbytes if total==0 else nbytes*scheduled//total
            self.coverage_preparation_products.append((done,produced))
        elif total==0 and not self.coverage_preparation_products:
            self.coverage_preparation_products.append((self.cycle,
                self.coverage_preparation_meta['link_bytes']))
        limit=float('inf') if force else self.cycle
        while (self.coverage_preparation_products and
               self.coverage_preparation_products[0][0]<=limit):
            ready,available=self.coverage_preparation_products.pop(0)
            self.coverage_preparation_available_ready=max(
                self.coverage_preparation_available_ready,ready)
            self.coverage_preparation_available_bytes=max(
                self.coverage_preparation_available_bytes,available)
            final=(self.coverage_preparation_scheduled_vector==total and
                   not self.coverage_preparation_products)
            while (self.coverage_preparation_written_bytes+DMA_BURST_BYTES<=
                   self.coverage_preparation_available_bytes):
                returned=self.memory.core.transaction(
                    max(self.cycle,self.coverage_preparation_available_ready),True,
                    self.memory.state,self.memory.pending,self.memory.counts,
                    self.memory.burst_cycles,self.c['dram_latency_cycles'],DMA_BURST_BYTES,
                    self.memory.dma_activity)
                self.coverage_preparation_written_bytes+=DMA_BURST_BYTES
                self.coverage_preparation_memory_ready=max(
                    self.coverage_preparation_memory_ready,int(returned))
            if final and self.coverage_preparation_written_bytes<self.coverage_preparation_available_bytes:
                returned=self.memory.core.transaction(
                    max(self.cycle,self.coverage_preparation_available_ready),True,
                    self.memory.state,self.memory.pending,self.memory.counts,
                    self.memory.burst_cycles,self.c['dram_latency_cycles'],DMA_BURST_BYTES,
                    self.memory.dma_activity)
                self.coverage_preparation_written_bytes=self.coverage_preparation_available_bytes
                self.coverage_preparation_memory_ready=max(
                    self.coverage_preparation_memory_ready,int(returned))
        complete=(self.coverage_preparation_scheduled_vector==total and
                  not self.coverage_preparation_products and
                  self.coverage_preparation_written_bytes>=self.coverage_preparation_meta['link_bytes'])
        if complete:
            self.coverage_preparation_ready=max(
                self.coverage_preparation_memory_ready,self.coverage_preparation_vector_ready)
            self.stats['coverage_preparation_cycles']=(
                self.coverage_preparation_ready-self.coverage_preparation_start)

    def finish_coverage_preparation(self):
        while self.coverage_preparation_ready is None:
            self.service_coverage_preparation(force=True)

    def initialize_responses(self,entries):
        """Clear value/valid pairs through the existing finite control ports.

        experiment reserves these bytes from the shared control capacity
        before the first ready window is admitted. No cold data DMA is needed
        for invocation-local invalid state.
        """
        from .predictor import response_record_words
        fields=response_record_words(self.c)
        words=entries*fields;cost=fields*math.ceil(entries/self.c['cam_lookups_per_cycle'])
        self.cycle=max(self.cycle,self.memory.clock)+cost
        self.memory.clock=self.cycle
        self.stats['response_summary_bytes']=words*4
        self.stats['control_sram_write_bytes']+=words*4
        self.stats['response_initialization_cycles']+=cost

    def vector_cost(self,event,count):
        if 'scalar_ops_per_row' in event:
            return max(math.ceil(count*event['scalar_ops_per_row']/self.c['vector_lanes']),
                       event.get('scalar_dependency_cycles',0) if count else 0)
        if (event.get('op')=='aten._unique2.default' and
                len(event['inputs'][0]['shape'])==1 and
                event['inputs'][0].get('dtype','').startswith(('torch.int','torch.uint','torch.bool'))):
            raise ValueError('Recapture integer unique with finite control metadata; generic timing is incomplete')
        from .reductions import vector_cycles as reduction_cycles
        cost=reduction_cycles(event,count,self.c['vector_lanes'])
        if cost is not None:return cost
        from .normalization_timing import current_statistics
        if current_statistics(event):
            # Statistics have already completed on the whole source. This
            # datapath only subtracts, rescales and applies optional affine.
            return (2+len(event['inputs'])-1)*math.ceil(count*event['outputs'][0]['shape'][-1]/self.c['vector_lanes'])
        if event['kind'] in ('geometry','coordinate_map','coordinate_decomposition'):return math.ceil(event['geometry_scalar_ops']/self.c['vector_lanes'])
        shape=event['outputs'][0]['shape'];width=shape[-1] if shape else 1
        op=event.get('op','');elements=count*width
        if event['kind']=='progressive_center_distance':
            width=event['inputs'][0]['shape'][-1];lanes=self.c['vector_lanes']
            # Promotion, subtraction and square; W-1 additions and one sqrt.
            # A dependent tree drains after the lane work; no wide temporary
            # is materialized and no additional vector unit is assumed.
            passes=2+int(event['inputs'][0]['element_bytes']==2)
            return (math.ceil(passes*count*width/lanes)+math.ceil(count*(width-1)/lanes)
                    +math.ceil(count/lanes)+math.ceil(math.log2(max(1,width))))
        if event['kind']=='progressive_difference':return math.ceil(2*elements/self.c['vector_lanes'])
        if event['kind']=='progressive_update':
            passes=1+int(event.get('materialize',True))+int(event.get('half_observation',True))
            if event['inputs'][0]['element_bytes']==2:passes+=2  # FP16 load promotion and rounded store.
            return math.ceil(passes*elements/self.c['vector_lanes'])
        if event['kind']=='progressive_observe':
            operations=count*(9*width+2*(event['bins']-1)+event['window']+8)
            return math.ceil(operations/self.c['vector_lanes'])
        if event['kind']=='ancestor_merge':return math.ceil(count*(60+21*width)/self.c['vector_lanes'])
        # Count elementary dependency stages of actual operations. These are
        # microcode costs, not fitted latency constants.
        if 'layer_norm' in op or 'batch_norm' in op:passes=NORM_PASSES
        elif 'gelu' in op:passes=GELU_PASSES
        elif 'vector_norm' in op:passes=VECTOR_NORM_PASSES
        elif 'sort' in op or 'unique' in op:passes=max(1,math.ceil(math.log2(max(2,elements))))
        else:passes=1
        if op.startswith(('aten.linalg_vector_norm','aten.var_mean','aten.sum.',
                          'aten.amax.','aten.argmax.','aten.topk.')):
            source=event['inputs'][0]
            # Reductions consume input elements even when their output is a
            # one-dimensional row-statistic vector. Output size is not work.
            elements=max(elements,math.ceil(math.prod(source['shape'])*count/max(1,rows(event['outputs'][0]))))
            if 'var_mean' in op:passes=NORM_PASSES
        return math.ceil(elements/self.c['vector_lanes'])*passes

    def output_transfer(self,nbytes,earliest,write=False,offset=None,*,staging_disjoint=False):
        from .feeder import window_transfer
        m=self.memory
        if offset is None:offset=self.accumulator_base(4)
        if write and self.attention_staging_storage_bytes and staging_disjoint:
            # The admitted Attention microprogram proves Q/score/AV bounds
            # once. This is a model assertion of that static opcode contract,
            # not a free general-purpose runtime range query.
            if any(lease is not None and offset<lease[1]+lease[2] and lease[1]<offset+nbytes
                   for lease in self.attention_staging_leases):
                raise ValueError('Attention scratch opcode overwrites retained K/V')
        elif write and self.attention_staging_storage_bytes:
            live=sum(lease is not None for lease in self.attention_staging_leases)
            if live:
                # Read each live range (base/size: four 32-bit words) on
                # existing control ports, compare, and update its valid bit.
                cost=math.ceil(live*4/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']+1
                earliest=max(earliest,m.clock)+cost
                m.clock+=cost
                self.stats['attention_coverage_staging_control_cycles']+=cost
            for owner,lease in enumerate(self.attention_staging_leases):
                if lease is not None and offset<lease[1]+lease[2] and lease[1]<offset+nbytes:
                    self.attention_staging_leases[owner]=None
                    self.stats['attention_coverage_staging_invalidations']+=1
        banks,calendar,lengths=m.ports(2)
        done=window_transfer(offset,nbytes,earliest,m.state,banks,calendar,lengths,
            self.c['bank_bytes_per_cycle'],m.counts,write)
        self.stats['output_window_write_bytes' if write else 'output_window_read_bytes']+=nbytes
        return int(done)

    def materialized_diagnostic_source(self,event,ids,input_ready):
        # One existing output owner holds the FP32 concatenation.
        # Mean, centering and norm follow the original arithmetic
        # order; only intermediate placement changes. The norm
        # reduction may consume the centered stream as it is
        # written, while normalization rereads that saved stream.
        channels=event['source_channels'];elements=len(ids)*channels
        nbytes=elements*4;scalar_bytes=len(ids)*4
        scalar_offset=self.accumulator_base(4)+nbytes
        if nbytes+scalar_bytes>self.output_budget():raise ValueError('Diagnostic source scratch exceeds output owner')
        done=self.stage_diagnostic_source(event,ids,input_ready)
        operations=0
        def source_pass(ready,ops,write=False):
            nonlocal operations
            ready=self.output_transfer(nbytes,ready)
            cost=math.ceil(len(ids)*ops/self.c['vector_lanes'])
            operations+=cost
            ready=self.issue_vector(ready,cost)
            return self.output_transfer(nbytes,ready,write=True) if write else ready
        if event['source_centered']:
            done=source_pass(done,channels)  # row mean
            done=self.output_transfer(scalar_bytes,done,write=True,offset=scalar_offset)
            done=self.output_transfer(scalar_bytes,done,offset=scalar_offset)
            done=source_pass(done,3*channels,write=True)  # subtract, square/sum
        else:done=source_pass(done,2*channels)
        done=self.output_transfer(scalar_bytes,done,write=True,offset=scalar_offset)
        done=self.output_transfer(scalar_bytes,done,offset=scalar_offset)
        scalar=math.ceil(len(ids)/self.c['vector_lanes'])
        done=self.issue_vector(done,scalar);operations+=scalar  # norm clamp
        done=self.output_transfer(scalar_bytes,done,write=True,offset=scalar_offset)
        done=self.output_transfer(scalar_bytes,done,offset=scalar_offset)
        done=source_pass(done,channels)  # normalize to final store
        done=self.issue_vector(done,scalar);operations+=scalar  # bias one
        self.stats['vector_cycles']+=operations
        self.stats['diagnostic_source_fused_rows']+=len(ids)
        self.stats['diagnostic_source_scratch_peak_bytes']=max(
            self.stats['diagnostic_source_scratch_peak_bytes'],nbytes+scalar_bytes)
        return done

    def stage_diagnostic_source(self,event,selected,earliest):
        """Route real source returns directly into the existing output owner."""
        from .copyroute import mapping,scratch_transfer
        channels=event['source_channels'];destination=self.accumulator_base(4)
        offset=0;done=earliest;m=self.memory
        banks,calendar,lengths=m.ports(2)
        direct='diagnostic_source_rows' in event
        parts=event['inputs'][:-1] if direct else event['inputs']
        if direct:
            # Returned index words authorize the source addresses. Waiting at
            # the final vector join would permit premature feature requests.
            earliest=max(earliest,m.access(self.g.sectors(event['inputs'][-1],selected),ready=earliest))
            m.clock=max(m.clock,earliest);done=earliest
        for part,spec in enumerate(parts):
            element=spec['element_bytes'];width=spec['shape'][-1]
            if element not in ((2,4) if direct else (4,)):
                raise ValueError('Unsupported diagnostic source precision')
            source_rows=np.asarray(event['diagnostic_source_rows'][part])[selected] if direct else selected
            starts=self.g.address_starts(spec,source_rows)
            keys,pointer,targets=mapping(starts,width,spec['stride'][-1]*element,element,
                destination+offset*4,channels*4)
            sink=(keys,pointer,targets,self.copy_ready,banks,calendar,lengths,False)
            if self.copy_source_order is not None:
                m.clock=max(m.clock,int(self.copy_source_order.max()))
            copied=m.access(self.g.sectors(spec,source_rows),ready=earliest,copy_sink=sink)
            if self.copy_source_order is not None:
                self.copy_source_order[:]=np.maximum(self.copy_source_order,copied)
            done=max(done,copied)
            self.stats['output_window_write_bytes']+=len(targets)*SECTOR_BYTES
            self.stats['diagnostic_source_staged_bytes']+=len(selected)*width*element
            if element==2:
                # Raw halves occupy the beginning of each disjoint FP32 row
                # segment. Expand backwards, using at most one existing vector
                # register batch; no whole-window return payload or cast cache.
                lanes=self.c['vector_lanes'];batch=max(1,lanes//width)
                converted=max(copied,self.diagnostic_cast_ready,self.vector)
                for row in range(0,len(selected),batch):
                    count=min(batch,len(selected)-row)
                    for end in range(width,0,-lanes):
                        column=max(0,end-lanes);used=end-column
                        base=destination+row*channels*4+offset*4
                        converted,nread=scratch_transfer(base,count,column,used,channels*4,2,converted,
                            m.state,banks,calendar,lengths,self.c['bank_bytes_per_cycle'],m.counts,False)
                        cost=math.ceil(count*used/lanes)
                        self.stats['diagnostic_source_cast_peak_elements']=max(
                            self.stats['diagnostic_source_cast_peak_elements'],count*used)
                        converted=self.issue_vector(int(converted),cost)
                        self.stats['vector_cycles']+=cost
                        converted,nwrite=scratch_transfer(base,count,column,used,channels*4,4,converted,
                            m.state,banks,calendar,lengths,self.c['bank_bytes_per_cycle'],m.counts,True)
                        self.stats['output_window_read_bytes']+=nread
                        self.stats['output_window_write_bytes']+=nwrite
                done=max(done,int(converted))
                self.diagnostic_cast_ready=int(converted)
                self.stats['diagnostic_source_cast_elements']+=len(selected)*width
            offset+=width
        if offset!=channels:raise ValueError('Diagnostic source channel partition mismatch')
        if direct:self.stats['diagnostic_source_direct_rows']+=len(selected)
        return done

    def groups(self,index,selected,bm):
        event=self.g.events[index]
        if event['kind']=='sdpa' and self.c.get('finite_operand_feeder',False):
            sequence=event['outputs'][0]['shape'][-2]
            for head in np.unique(selected//sequence):
                yield from self._groups(index,selected[selected//sequence==head],bm)
        else:yield from self._groups(index,selected,bm)

    def coverage_chain_references(self,index,selected):
        """Return statically row-local inputs used later in this event chain."""
        result=[]
        if not self.c.get('coverage_chain_inputs',False):return result
        event=self.g.events[index]
        if len(event['outputs'])!=1:return result
        count=rows(event['outputs'][0]);produced={key(event['outputs'][0])}
        for following in self.g.events[index+1:]:
            if (following['kind'] in ('linear','spconv','ME','sdpa','attention') or
                following.get('op')=='aten.mm.default' or len(following['outputs'])!=1 or
                rows(following['outputs'][0])!=count):break
            external=[]
            for spec in following['inputs']:
                if spec['role']!='feature' or key(spec) in produced:continue
                if not self.can_reclaim(following,spec):
                    external=None
                    break
                external.append(spec)
            if external is None:break
            result.extend((spec,selected) for spec in external)
            produced.add(key(following['outputs'][0]))
        return result

    def coverage_references(self,index,selected):
        """Extend a B window through statically row-local residual inputs."""
        from .normalization_timing import references
        result=references(self.g,index,selected)
        chain=self.coverage_chain_references(index,selected)
        result.extend(chain)
        self.stats['coverage_chain_input_references']+=len(chain)*len(selected)
        return result

    def linked_frontier(self,window,records,examined=None,staging=None):
        """Intersect witnessed coverage consumers with same-version inputs.

        Host dictionaries represent only the admitted finite descriptors.
        Row/edge probes and sorted reference intersections pay the existing
        control lanes; no target future values or extra cache are consulted.
        Masks survive a peer's completion only in its still-pending consumer.
        """
        examined=set() if examined is None else examined
        incremental=self.c.get('coverage_link_incremental_pairs',False)
        local_membership=self.c.get('coverage_local_membership',False)
        owner={} if incremental else {int(row):i for i,(ids,_,_) in enumerate(window)
               if records[id(ids)][1] for row in ids}
        # The Python owner dict stands for a sorted row/group table, not a
        # free associative map. Count sort/move and binary-search word reads.
        depth=math.ceil(math.log2(max(2,len(owner))))
        work=2*len(owner)*depth
        ports=self.c['cam_lookups_per_cycle']
        control_cycles=2*math.ceil(len(owner)/ports)*depth
        neighbor_count=sum(len(records[id(ids)][0]) for ids,_,_ in window)
        # Keys search in parallel, but each key's next address depends on its
        # previous comparison. Two words carry the row/group table record.
        if not incremental:control_cycles+=2*math.ceil(neighbor_count/ports)*depth
        def membership_for(ids,record):
            nonlocal work,control_cycles
            if len(record)==4:
                if not record[1]:record.append(None);return None
                # Replace the sorted owner search with a bounded membership
                # bitmap in this group's already capacity-checked metadata.
                base_word=0
                if local_membership:
                    # A current descriptor already carries these immutable
                    # IDs. Read/reduce min/max and validate consecutive IDs;
                    # no geometry qualification or consumer test disappears.
                    work+=3*len(ids)+3
                    control_cycles+=3*math.ceil(len(ids)/ports)+2*math.ceil(math.log2(max(2,len(ids))))+3
                    low=int(ids.min()) if len(ids) else 0
                    high=int(ids.max())+1 if len(ids) else 0
                    if high-low==len(ids) and np.all(ids[1:]==ids[:-1]+1):
                        record.append((low,high,None))
                        self.stats['coverage_linked_interval_builds']+=1
                        return record[4]
                    base_word=low//32
                words=(int(ids.max())//32+1-base_word) if len(ids) else 0
                membership=np.zeros(words,np.uint32)
                work+=words;control_cycles+=math.ceil(words/ports)
                for begin in range(0,len(ids),ports):
                    keys=ids[begin:begin+ports];addresses=keys//32-base_word
                    work+=2*len(keys);control_cycles+=2
                    # The existing control SRAM maps words modulo lookup
                    # lanes. Distinct words in one bank also serialize; same
                    # word keys are not atomically combined for free.
                    conflicts=np.bincount(addresses%ports,minlength=ports)
                    rounds=int(conflicts.max()) if len(keys) else 0
                    work+=3*len(keys);control_cycles+=4+rounds*(self.c['cam_latency_cycles']+2)
                    self.stats['coverage_linked_bitmap_rmw_rounds']+=rounds
                    np.bitwise_or.at(membership,addresses,
                        np.left_shift(np.uint32(1),(keys%32).astype(np.uint32)))
                record.append((base_word*32,base_word*32+words*32,membership) if local_membership else membership)
                self.stats['coverage_linked_bitmap_builds']+=1
            return record[4]
        def overlap(neighbors,membership):
            nonlocal work,control_cycles
            if not len(neighbors) or membership is None:return False
            low=0;high=len(membership)*32 if not local_membership else membership[1]
            if local_membership:
                low,high,bits=membership
                # Read descriptor base/end/mode and test each actual link ID.
                work+=3;control_cycles+=math.ceil(3/ports)+self.c['cam_latency_cycles']
                if bits is None:
                    for begin in range(0,len(neighbors),ports):
                        keys=neighbors[begin:begin+ports]
                        words=math.ceil(keys.dtype.itemsize/4)
                        work+=len(keys)*(words+3)
                        control_cycles+=math.ceil(len(keys)*words/ports)+3+math.ceil(math.log2(max(2,len(keys))))
                        self.stats['coverage_linked_interval_probe_rows']+=len(keys)
                        if np.any((keys>=low)&(keys<high)):return True
                    return False
                membership=bits
            for begin in range(0,len(neighbors),ports):
                keys=neighbors[begin:begin+ports];addresses=(keys-low)//32
                # Read neighbor IDs and their real bitmap words; repeated
                # addresses still consume lookup bandwidth, not free CAMs.
                valid=(keys>=low)&(keys<high)&(addresses<len(membership))
                conflicts=np.bincount(addresses[valid]%ports,minlength=ports)
                rounds=int(conflicts.max()) if np.any(valid) else 0
                key_words=math.ceil(keys.dtype.itemsize/4)
                work+=len(keys)*(3+key_words)
                reads=math.ceil(len(keys)*key_words/ports)
                control_cycles+=reads+rounds+self.c['cam_latency_cycles']+2+math.ceil(math.log2(max(2,len(keys))))
                self.stats['coverage_linked_bitmap_read_rounds']+=rounds
                if np.any((membership[addresses[valid]]>>(keys[valid]%32).astype(np.uint32))&np.uint32(1)):return True
            return False
        def qualify(i,j):
            nonlocal work,control_cycles
            left=records[id(window[i][0])];right=records[id(window[j][0])]
            version_depth=math.ceil(math.log2(max(2,len(right[1])+1)))
            work+=len(left[1])*16*version_depth
            control_cycles+=math.ceil(len(left[1])/ports)*16*version_depth
            pieces=[]
            for version,(spec,source_rows) in left[1].items():
                other=right[1].get(version)
                if other is None:continue
                if self.c.get('coverage_compact_references',False):
                    from .coverage_references import intersection
                    shared,words,cycles=intersection(source_rows,other[1],ports)
                    work+=words;control_cycles+=cycles
                    self.stats['coverage_compact_reference_intersections']+=1
                else:
                    row_depth=math.ceil(math.log2(max(2,len(other[1])+1)))
                    work+=len(source_rows)*row_depth
                    control_cycles+=math.ceil(len(source_rows)/ports)*row_depth
                    shared=np.intersect1d(source_rows,other[1],assume_unique=True)
                if len(shared):pieces.append(self.g.sectors(spec,shared))
                if staging is not None and len(shared)==len(source_rows)==len(other[1]) and len(shared):
                    operands,permissions=staging
                    for operand,view in enumerate(operands):
                        if version==(key(view),self.coverage_rows.signature(view)):
                            # The existing native references contain the full
                            # actual K/V head. Both groups must share every
                            # input row, not merely their physical sectors.
                            for group in (window[i][0],window[j][0]):
                                permissions[id(group)]|=1<<operand
                            work+=4;control_cycles+=2*(self.c['cam_latency_cycles']+2)
            if pieces:
                shared=np.unique(np.concatenate(pieces))
                work+=len(left[2])+len(right[2])+len(shared)*4
                control_cycles+=math.ceil((len(left[2])+len(right[2])+len(shared)*4)/ports)
                left[2]=np.union1d(left[2],shared);right[2]=np.union1d(right[2],shared)
                self.stats['coverage_linked_shared_pairs']+=1
            self.stats['coverage_linked_candidate_pairs']+=1
        if incremental:
            for i,(ids,_,_) in enumerate(window):
                left=records[id(ids)]
                for j in range(i+1,len(window)):
                    right=records[id(window[j][0])]
                    a,b=sorted((left[3],right[3]))
                    work+=1;control_cycles+=1
                    if (a,b) in examined:
                        self.stats['coverage_linked_cached_pairs']+=1
                        continue
                    # The CSR payload lengths are already in admitted
                    # descriptors. No incoming probe means no bitmap read
                    # or construction; reserved bytes are not refunded.
                    linked=(bool(len(left[0])) and overlap(left[0],membership_for(window[j][0],right))) or (
                            bool(len(right[0])) and overlap(right[0],membership_for(ids,left)))
                    if linked:qualify(i,j)
                    # Cache negative geometry checks as well as completed
                    # input qualification, until either stable slot retires.
                    examined.add((a,b));work+=1;control_cycles+=2
        else:
            for i,(ids,_,_) in enumerate(window):
                neighbors=records[id(ids)][0];work+=len(neighbors)*depth
                for neighbor in neighbors:
                    j=owner.get(int(neighbor))
                    if j is None or i==j:continue
                    # A miss is processed immediately before its checked bit
                    # is set. There is no second pair set, queue or bitmap.
                    a,b=sorted((records[id(ids)][3],records[id(window[j][0])][3]))
                    work+=1;control_cycles+=1
                    if (a,b) not in examined:
                        qualify(min(i,j),max(i,j))
                        examined.add((a,b));work+=1;control_cycles+=2
                    else:self.stats['coverage_linked_cached_pairs']+=1
        masks=[records[id(ids)][2] for ids,_,_ in window]
        # Read masks once more to construct the directory-protection front.
        work+=sum(map(len,masks))
        control_cycles+=math.ceil(sum(map(len,masks))/ports)
        cost=control_cycles+self.c['cam_latency_cycles']
        self.memory.clock+=cost
        self.stats['coverage_linked_control_cycles']+=cost
        self.stats['coverage_linked_control_words']+=work
        protected=np.unique(np.concatenate(masks))
        self.stats['coverage_linked_protected_sectors']+=len(protected)
        return protected

    def _groups(self,index,selected,bm):
        if not self.coverage_phase_active or self.private_coverage_rows(self.g.events[index]):
            if self.coverage_phase_active:self.stats['coverage_private_bypass_rows']+=len(selected)
            for start in range(0,len(selected),bm):yield selected[start:start+bm]
            return
        if 0<len(selected)<=bm:
            # A single admitted group has no peer whose shared inputs can be
            # retained or whose issue order can change.  Send it through the
            # ordinary finite feeder without constructing a coverage window.
            self.stats['coverage_single_group_bypass_rows']+=len(selected)
            yield selected
            return
        cursor=0;window=[];metadata_bytes=0;linked_records={};examined=set()
        staging_permissions={}
        staging=(self.g.events[index]['inputs'][1:3],staging_permissions) if (
            self.attention_staging_storage_bytes and self.g.events[index]['kind']=='sdpa') else None
        boundary=self.c.get('coverage_shared_boundary',False)
        linked=self.c.get('coverage_linked_supply',False)
        if linked:
            clear=math.ceil(self.coverage_link_state_bytes/(4*self.c['cam_lookups_per_cycle']))
            self.memory.clock+=clear;self.stats['coverage_linked_control_cycles']+=clear
        capacity=min(self.c['feature_sram_bytes']//SLOT_BYTES,self.c['cam_entries'])
        consumed=set()
        while cursor<len(selected) or window:
            while cursor<len(selected) and len(window)<self.c['queue_entries']:
                ids=selected[cursor:cursor+bm]
                refs=self.coverage_references(index,ids)
                pieces=[self.g.sectors(spec,source_rows) for spec,source_rows in refs if spec['role']=='feature']
                sectors=np.unique(np.concatenate(pieces)) if pieces else np.empty(0,dtype=np.int64)
                tag_count=len(np.unique(sectors//SLOT_SECTORS))
                if boundary:
                    # A rejected tentative candidate is not kept in a free
                    # lookahead cache. Its tag scan is paid again if retried;
                    # actual admission still pays the retained descriptor.
                    cost=math.ceil(tag_count/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
                    self.memory.clock+=cost
                    self.stats['coverage_admission_tag_checks']+=tag_count
                    self.stats['coverage_admission_cycles']+=cost
                if boundary and window:
                    tags=np.unique(np.concatenate([s//SLOT_SECTORS for _,s,_ in window]+[sectors//SLOT_SECTORS]))
                    if len(tags)>capacity:
                        self.stats['coverage_capacity_boundaries']+=1
                        break
                # Shared probing keeps a tag/mask snapshot and the existing
                # per-group reference plus score. Reserve the conservative
                # per-edge bound, even when different groups share a tag.
                size=tag_count*(32 if boundary or self.c.get('coverage_shared_probe',False) else 16)
                if self.c.get('coverage_shared_probe',False):size+=8
                if linked:
                    query_work=self.coverage_rows.query_work.copy()
                    if self.g.events[index].get('diagnostic_affine_scatter',False):
                        # Compact consumer ordinals are not destination
                        # native IDs. Until this partial-update domain has a
                        # proven geometry binding, do not guess its links.
                        empty=np.empty(0,np.int64);neighbors,nbytes,bursts=empty,0,(empty,empty)
                    else:
                        neighbors,nbytes,bursts=self.coverage_rows.neighbors(self.g.events[index]['outputs'][0],ids,
                            deduplicate=not self.c.get('coverage_link_incremental_pairs',False))
                    query_work=self.coverage_rows.query_work-query_work
                    if query_work.any():
                        # Offset DMA addresses depend on selected->token
                        # conversion. It must acquire the actual vector unit
                        # before either dependent CSR transaction is issued.
                        control=math.ceil(int(query_work[0])/self.c['cam_lookups_per_cycle'])
                        vector=math.ceil(int(query_work[0])*4/self.c['vector_lanes'])
                        self.memory.clock=self.issue_vector(self.memory.clock+control,vector)
                        self.stats['vector_cycles']+=vector
                        self.stats['coverage_attention_query_control_cycles']+=control
                        self.stats['coverage_attention_query_vector_cycles']+=vector
                    # Metadata uses the real shared DMA, including latency,
                    # bursts and contention, rather than an ideal byte ratio.
                    ready=self.memory.clock
                    for phase in bursts:
                        # Neighbor addresses are not authorized until their
                        # offsets have returned. Distinct discontiguous CSR
                        # bursts remain distinct transactions.
                        issued=ready
                        for _ in phase:
                            ready=max(ready,self.memory.core.transaction(issued,False,
                                self.memory.state,self.memory.pending,self.memory.counts,
                                self.memory.burst_cycles,self.c['dram_latency_cycles'],DMA_BURST_BYTES,
                                self.memory.dma_activity))
                    self.memory.clock=ready+math.ceil(nbytes/(4*self.c['cam_lookups_per_cycle']))
                    if query_work.any():
                        control=sum(math.ceil(int(words)/self.c['cam_lookups_per_cycle']) for words in query_work[1:3])
                        vector=math.ceil((int(query_work[3])-int(query_work[0])*4)/self.c['vector_lanes'])
                        self.memory.clock=self.issue_vector(self.memory.clock+control,vector)
                        self.stats['vector_cycles']+=vector
                        self.stats['coverage_attention_query_control_cycles']+=control
                        self.stats['coverage_attention_query_vector_cycles']+=vector
                        self.stats['coverage_attention_query_rows']+=len(ids)
                    self.stats['coverage_linked_index_read_bytes']+=nbytes
                    self.stats['coverage_linked_index_bursts']+=sum(map(len,bursts))
                    versioned={}
                    for spec,source_rows in refs:
                        if spec['role']=='feature':
                            version=(key(spec),self.coverage_rows.signature(spec))
                            previous=versioned.get(version,(spec,np.empty(0,np.int64)))[1]
                            versioned[version]=(spec,np.union1d(previous,source_rows))
                    # Per-version tag/mask/carry plus finite native-row lookup
                    # and neighbor IDs; all share the remaining control SRAM.
                    size=max(size,sum(len(np.unique(self.g.sectors(s,r)//SLOT_SECTORS))
                                      for s,r in versioned.values())*32)
                    size+=len(ids)*8+neighbors.nbytes+len(versioned)*64
                    if self.c.get('coverage_compact_references',False):
                        from .coverage_references import prepare,storage_bytes,RowInterval
                        for version,(spec,source_rows) in list(versioned.items()):
                            # The compiler's original per-group reference
                            # program is a burst-aligned 32-bit row list.
                            # Compression cannot erase the input of its own
                            # validation pass or borrow the feature payload's
                            # later reads. Stream it through the real DMA.
                            read_bytes=len(source_rows)*4
                            issued=ready=self.memory.clock
                            for _ in range(math.ceil(read_bytes/DMA_BURST_BYTES)):
                                ready=max(ready,self.memory.core.transaction(issued,False,
                                    self.memory.state,self.memory.pending,self.memory.counts,
                                    self.memory.burst_cycles,self.c['dram_latency_cycles'],DMA_BURST_BYTES,
                                    self.memory.dma_activity))
                            self.memory.clock=ready
                            self.stats['coverage_compact_reference_validation_bytes']+=read_bytes
                            reference,words,cycles=prepare(source_rows,self.c['cam_lookups_per_cycle'])
                            versioned[version]=(spec,reference)
                            self.memory.clock+=cycles
                            self.stats['coverage_linked_control_cycles']+=cycles
                            self.stats['coverage_linked_control_words']+=words
                            self.stats['coverage_compact_reference_preparation_cycles']+=cycles
                            self.stats['coverage_compact_reference_intervals']+=int(isinstance(reference,RowInterval))
                            size+=storage_bytes(reference)
                    else:size+=sum(len(r)*4 for _,r in versioned.values())
                    if self.c.get('coverage_link_incremental_pairs',False):
                        if self.c.get('coverage_local_membership',False):
                            low=int(ids.min()) if len(ids) else 0
                            high=int(ids.max())+1 if len(ids) else 0
                            contiguous=high-low==len(ids) and np.all(ids[1:]==ids[:-1]+1)
                            words=0 if contiguous or not len(ids) else (high-1)//32-low//32+1
                            size+=12+words*4
                            # The representation must be selected before
                            # admission; charge these same bounded ID scans.
                            cost=3*math.ceil(len(ids)/self.c['cam_lookups_per_cycle'])+2*math.ceil(math.log2(max(2,len(ids))))+3
                            self.memory.clock+=cost
                            self.stats['coverage_linked_control_cycles']+=cost
                        else:size+=(int(ids.max())//32+1)*4 if len(ids) else 0
                    if staging is not None:size+=4  # two K/V permission bits in one SRAM word
                if window and metadata_bytes+size>self.control_capacity():break
                if size>self.control_capacity():
                    # Oversized tasks cannot have all their references in the
                    # control table. They retain normal LRU and issue alone.
                    sectors=np.empty(0,dtype=np.int64);size=0
                    if linked:
                        neighbors=np.empty(0,np.int64);versioned={}
                        self.stats['coverage_linked_oversized_groups']+=1
                window.append((ids,sectors,size));metadata_bytes+=size;cursor+=len(ids)
                if linked:
                    used={record[3] for record in linked_records.values()}
                    slot=next(slot for slot in range(self.c['queue_entries']) if slot not in used)
                    linked_records[id(ids)]=[neighbors,versioned,np.empty(0,np.int64),slot]
                    if staging is not None:staging_permissions[id(ids)]=0
                self.stats['scheduler_metadata_bytes']+=size
                if linked:self.stats['scheduler_metadata_peak_bytes']=max(self.stats['scheduler_metadata_peak_bytes'],metadata_bytes)
                self.memory.clock+=math.ceil(size*self.c['clock_hz']/self.c['dram_bandwidth_bytes_per_second'])
                if boundary and tag_count>capacity:
                    self.stats['coverage_oversized_groups']+=1
                    break
            if len(window)==1:
                # No coverage peer remains in this finite window.  The
                # admitted group keeps all charged metadata and data traffic,
                # but link construction and candidate ranking have no action
                # to take and can retire immediately.
                ids,_,charged_size=window.pop()
                metadata_bytes-=charged_size
                if linked:
                    if staging is not None:self.attention_staging_permission=staging_permissions.pop(id(ids))
                    linked_records.pop(id(ids))
                self.memory.protect(np.empty(0,dtype=np.int64))
                self.stats['coverage_single_group_bypass_rows']+=len(ids)
                yield ids
                continue
            if linked:
                protected=self.linked_frontier(window,linked_records,examined,staging)
                self.memory.protect(protected)
                shared_tags=np.unique(protected//SLOT_SECTORS)
                self.stats['coverage_shared_peak_slots']=max(self.stats['coverage_shared_peak_slots'],len(shared_tags))
            elif boundary:
                # One reference per ready group, rather than per sector of a
                # wide private row. Past consumers extend protection only to
                # the actual remaining consumer of this admitted boundary.
                references=[np.unique(s//SLOT_SECTORS) for _,s,_ in window]
                tags,counts=np.unique(np.concatenate(references),return_counts=True)
                consumed.intersection_update(map(int,tags))
                shared=(counts>1)|np.isin(tags,list(consumed))
                shared_tags=tags[shared]
                all_sectors=np.concatenate([s for _,s,_ in window])
                protected=all_sectors[np.isin(all_sectors//SLOT_SECTORS,shared_tags)]
                updates=sum(len(t) for t in references)+len(tags)
                cost=math.ceil(updates/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
                self.memory.clock+=cost
                self.stats['coverage_reference_cycles']+=cost
                self.stats['coverage_reference_updates']+=updates
                if len(tags)<=capacity:
                    self.stats['coverage_boundary_peak_slots']=max(self.stats['coverage_boundary_peak_slots'],len(tags))
                self.stats['coverage_shared_peak_slots']=max(self.stats['coverage_shared_peak_slots'],len(shared_tags))
                self.stats['coverage_private_unprotected_slots']+=int((~shared).sum())
                self.memory.protect(np.unique(protected))
            else:self.memory.protect(np.concatenate([s for _,s,_ in window]))
            scores=[]
            # Admission retains the finite coverage-ordered ready window.
            # A bounded front shortlist trades global rescoring for fewer
            # directory probes; it never observes future cache accesses.
            score_limit=self.c.get('coverage_score_candidates',self.c['queue_entries'])
            if score_limit<=0:raise ValueError('Coverage scorer must examine at least one ready group')
            candidates=window[:score_limit]
            if linked:
                # Score only geometrically qualified same-version input
                # masks. Removing links therefore removes this privilege.
                candidates=[(ids,linked_records[id(ids)][2],size) for ids,_,size in candidates]
            if len(candidates)==1 and self.c.get('skip_single_candidate_scores',False):
                # There is no alternative to rank. Admission and residency
                # maintenance above still execute with their normal costs.
                scores=[(0,0)]
                self.stats['scheduler_single_choice_skips']+=1
            elif self.c.get('coverage_shared_probe',False):
                requests=[]
                for _,sectors,_ in candidates:
                    tags=np.unique(sectors//SLOT_SECTORS);masks=np.zeros(len(tags),np.uint64)
                    np.bitwise_or.at(masks,np.searchsorted(tags,sectors//SLOT_SECTORS),
                        np.left_shift(np.uint64(1),(sectors%SLOT_SECTORS).astype(np.uint64)))
                    requests.append((tags,masks))
                tags=np.unique(np.concatenate([t for t,_ in requests]))
                valid=self.memory.core.resident_tag_masks(tags,self.memory.directories[0],self.memory.slots[0])
                references=sum(len(t) for t,_ in requests)
                # SWAR popcount needs 17 scalar operations, preceded by the
                # requested/resident AND; reductions use the same lanes.
                popcount_and_mask_ops=18
                arithmetic=popcount_and_mask_ops*math.ceil(references/self.c['vector_lanes'])
                for (_,sectors,_),(group_tags,masks) in zip(candidates,requests):
                    hit=int(self.memory.core.masked_hits(masks,valid[np.searchsorted(tags,group_tags)]))
                    requested=len(window[len(scores)][1]) if linked else len(sectors)
                    scores.append((hit,-(requested-hit)))
                    reduction=len(group_tags)
                    while reduction>1:
                        reduction=math.ceil(reduction/2)
                        arithmetic+=math.ceil(reduction/self.c['vector_lanes'])
                ports=self.c['cam_lookups_per_cycle']
                cost=math.ceil(len(tags)/ports)+self.c['cam_latency_cycles']
                cost+=math.ceil(references*16/(ports*4))+math.ceil(len(candidates)/ports)+arithmetic
                if self.c.get('vector_ready_queue',False):
                    control_cost=cost-arithmetic
                    self.memory.clock=self.issue_vector(self.memory.clock+control_cost,arithmetic)
                else:
                    self.memory.clock=self.issue_vector(self.memory.clock,cost)
                self.stats['vector_cycles']+=arithmetic
                self.stats['scheduler_cam_lookups']+=len(tags)
                self.stats['scheduler_shared_probe_references']+=references
                self.stats['scheduler_control_read_bytes']+=references*16
            else:
                for ids,sectors,_ in candidates:
                    hit=self.memory.resident(sectors)
                    requested=len(window[len(scores)][1]) if linked else len(sectors)
                    scores.append((hit,-(requested-hit)))
                    lookups=len(np.unique(sectors//SLOT_SECTORS))
                    self.memory.clock+=math.ceil(lookups/self.c['cam_lookups_per_cycle'])
                    self.stats['scheduler_cam_lookups']+=lookups
            self.stats['scheduler_scored_candidates']+=len(candidates)
            chosen=max(range(len(candidates)),key=lambda j:scores[j])
            ids,sectors,charged_size=window.pop(chosen);metadata_bytes-=charged_size
            if linked:
                if staging is not None:self.attention_staging_permission=staging_permissions.pop(id(ids))
                slot=linked_records.pop(id(ids))[3]
                examined.difference_update(pair for pair in tuple(examined) if slot in pair)
                # Clear this fixed bitmap row/column before slot reuse. Old
                # feature masks remain only in actual pending consumers.
                clear_cycles=math.ceil(2*self.c['queue_entries']/self.c['cam_lookups_per_cycle'])
                self.memory.clock+=clear_cycles
                self.stats['coverage_linked_control_cycles']+=clear_cycles
            if boundary:
                if not linked:consumed.update(map(int,np.unique(sectors//SLOT_SECTORS)))
                prefetch=self.c.get('coverage_prefetch_groups',0)
                if not isinstance(prefetch,int) or not 0<=prefetch<=self.c['queue_entries']:
                    raise ValueError('Coverage prefetch must fit the finite ready queue')
                for future_ids,future,_ in window[:prefetch]:
                    shared_sectors=(linked_records[id(future_ids)][2] if linked else
                                    future[np.isin(future//SLOT_SECTORS,shared_tags)])
                    if len(shared_sectors):
                        self.memory.access(shared_sectors,prefetch=True)
                        self.stats['coverage_prefetch_requested_bytes']+=len(shared_sectors)*SECTOR_BYTES
            yield ids
        self.memory.protect(np.empty(0,dtype=np.int64))

    def chain_weight_bytes(self,event):
        """Scheduling footprint; actual fills still use the finite directory.

        A linear already lowered into N stripes need not reserve its entire
        frozen matrix just to join a row-local chain. This is a grouping
        heuristic, not a claim that logical stripe bytes imply physical hits.
        Every strided weight load, slot fill and eviction remains executed.
        """
        weights=[s for s in event['inputs'] if s['role']=='weight']
        total=sum(s['bytes'] for s in weights)
        width=self.c.get('dense_output_columns',0)
        if (not self.c.get('row_chain_weight_stripes',False) or not width
            or not self.c.get('finite_operand_feeder',False)
            or event['kind']!='linear' or len(event['inputs'])<2):return total
        matrix=event['inputs'][1]
        if (matrix['role']!='weight' or len(matrix['shape'])!=2
            or key(matrix) in self.produced_versions):return total
        if self.c.get('dense_columns_oversized_only',False) and matrix['bytes']<=self.c['weight_sram_bytes']:
            return total
        stripe=min(width,matrix['shape'][0])*matrix['shape'][1]*matrix['element_bytes']
        return total-matrix['bytes']+stripe

    def schedule(self,live,control,start=0,end=None,allow_wavefront=True):
        allowed=('aten.add.','aten.add_.','aten.sub.','aten.mul.','aten.div.','aten.relu',
                 'aten.gelu','aten.layer_norm','aten.to.','aten.clamp_min','aten.linalg_vector_norm')
        checkpoints=self.g.maps.get('checkpoints',[])
        across_metadata=self.c.get('stream_across_stage_metadata',False)
        if across_metadata:
            # Labels do not add a dependency to the mathematical DAG. Only
            # observations actually executed by A require a complete window.
            checkpoints=(checkpoints[self.g.maps.get('observation_begin',0):-1]
                         if control is not None else [])
        boundaries={cp['after_event'] for cp in checkpoints}
        if control is not None and control[3]=='pilot' and uses_witness_selector(self.c):
            boundaries.add(self.g.maps['response_context']['after_event'])
        boundaries.update(self.g.maps.get('diagnostic_control_barriers',{}))
        boundaries.update(self.checkpoint_fork.boundaries())
        end=len(self.g.events) if end is None else end
        patch_regions={}
        if allow_wavefront and self.c.get('attention_patch_wavefront',False):
            from .wavefront import regions
            patch_regions=regions(self.g,boundaries)
            if self.c.get('attention_patch_oversized_only',False):
                capacity=min(self.c['feature_sram_bytes'],self.c['cam_entries']*SLOT_BYTES)
                eligible={at:r for at,r in patch_regions.items() if
                    max(e['outputs'][0]['bytes'] for e in self.g.events[r['start']:r['end']+1])>capacity}
                self.stats['small_attention_regions_skipped']+=len(patch_regions)-len(eligible)
                patch_regions=eligible
        def independent(e):
            return len(e['outputs'])==1 and len(e['outputs'][0]['shape'])==2 and (e['kind']=='linear' or e.get('op','').startswith(allowed)
                or (e.get('diagnostic_row_local',False) and e['kind']!='indexed_store')
                or (self.c.get('stream_ancestor_epilogues',False) and e['kind']=='ancestor_merge')
                or (self.c.get('stream_ancestor_epilogues',False) and e.get('op')=='aten.copy_.default'
                    and self.g.trace.get('diagnostic_observations') is False))
        def matching_rows(e,producers):
            for position,source in enumerate(e['inputs']):
                previous=producers.get(key(source))
                if previous is not None and 'diagnostic_control_rows' in e:
                    mapping=np.asarray(e['diagnostic_control_rows'][position])
                    if not np.array_equal(mapping,np.arange(rows(e['outputs'][0]))):return False
                # A matrix RHS consumes its whole K/N domain, regardless of
                # the output row window. It cannot read a partial producer.
                if previous is not None and (source['role']=='weight' or
                    (e['kind']=='linear' and position==1)):return False
                if previous is not None and e.get('op')=='aten.index.Tensor' and source is e['inputs'][0]:
                    at=next(i for i,event in enumerate(self.g.events) if event is e)
                    mapping=self.g.maps['indices'][at][0]
                    if len(mapping)!=rows(source) or not np.array_equal(mapping,np.arange(len(mapping))):return False
                if previous is not None and e.get('indexed_source',False) and position==0:
                    at=next(i for i,event in enumerate(self.g.events) if event is e)
                    mapping=np.asarray(self.g.maps['indices'][at][0])
                    if len(mapping)!=rows(source) or not np.array_equal(mapping,np.arange(len(mapping))):return False
                if (previous is not None and e['kind']=='ancestor_merge' and
                    (position<5 or 10<=position<20)):
                    at=next(i for i,event in enumerate(self.g.events) if event is e)
                    mapping=self.g.maps['indices'][at][position%5]
                    # Equal row counts do not imply a row-local ancestor.
                    # A producer window can only feed this merge window when
                    # the actual native map preserves every row identity.
                    if len(mapping)!=rows(source) or not np.array_equal(mapping,np.arange(len(mapping))):return False
                if previous is not None and any(source[field]!=previous[field] for field in ('address','shape','stride','element_bytes')):
                    # A row statistic may be broadcast across this same row.
                    # It cannot supply another row, a shifted view or a
                    # general reshape before the producer window completes.
                    broadcast=(self.c.get('stream_broadcast_epilogues',False) and
                        source['address']==previous['address'] and
                        source['element_bytes']==previous['element_bytes'] and
                        len(source['shape'])==len(previous['shape'])==2 and
                        previous['shape'][-1]==1 and source['stride'][-1]==0 and
                        source['shape'][:-1]==previous['shape'][:-1] and
                        source['stride'][:-1]==previous['stride'][:-1])
                    if not broadcast:return False
            return True
        i=start
        while i<end:
            self.chain_floor=None;self.chain_end=True
            # Direction derives from the existing row-window loop counter;
            # it is not a persistent per-graph weight-owner table.
            self.chain_window_ordinal=0
            if i in self.checkpoint_fork.starts and self.checkpoint_fork.starts[i]<end:
                group=self.checkpoint_fork.window_rows(self,i,live)
                if group is not None:
                    yield from self.checkpoint_fork.schedule(self,i,live,group)
                    i=self.checkpoint_fork.starts[i]+1
                    continue
            if (i in self.integer_mask_fusions and i+1<end and i not in boundaries and
                np.array_equal(live[i],live[i+1])):
                original=self.g.events[i]
                target=self.integer_mask_fusions[i]
                fused=dict(original,outputs=[target],op='gadra.bitwise_and_ne_fused',
                           scalar_ops_per_row=2*target['shape'][-1],
                           integer_mask_fusion=True,
                           integer_mask_elided_bytes_per_row=(
                               original['outputs'][0]['shape'][-1]
                               * original['outputs'][0]['element_bytes']))
                yield i,fused,live[i],True
                yield i+1,dict(self.g.events[i+1],integer_mask_fusion_join=True),live[i+1],True
                i+=2
                continue
            if (i in self.training_batch_norm_relu_fusions and i+1<end and
                i not in boundaries and np.array_equal(live[i],live[i+1])):
                original=self.g.events[i]
                fused=dict(original,outputs=[self.training_batch_norm_relu_fusions[i]],
                           training_batch_norm_relu_fusion=True)
                yield i,fused,live[i],True
                yield i+1,dict(self.g.events[i+1],
                               training_batch_norm_relu_fusion_join=True),live[i+1],True
                i+=2
                continue
            if (i in self.training_batch_norm_add_relu_fusions and i+2<end and
                i not in boundaries and i+1 not in boundaries and
                np.array_equal(live[i],live[i+1]) and
                np.array_equal(live[i],live[i+2])):
                original=self.g.events[i]
                target,residual=self.training_batch_norm_add_relu_fusions[i]
                fused=dict(original,inputs=original['inputs']+[residual],outputs=[target],
                           normalization_row_input_positions=(len(original['inputs']),),
                           training_batch_norm_add_relu_fusion=True)
                yield i,fused,live[i],True
                yield i+1,dict(self.g.events[i+1],
                    training_batch_norm_add_relu_fusion_join='add'),live[i+1],True
                yield i+2,dict(self.g.events[i+2],
                    training_batch_norm_add_relu_fusion_join='relu'),live[i+2],True
                i+=3
                continue
            if i in patch_regions and patch_regions[i]['end']<end:
                from .wavefront import schedule
                if (yield from schedule(self,patch_regions[i],live,control)):
                    i=patch_regions[i]['end']+1
                    continue
            chain=[i];event=self.g.events[i]
            if (i in self.writeback_casts and i+1<end and i not in boundaries and
                np.array_equal(live[i],live[i+1])):
                fused=dict(event,outputs=[self.writeback_casts[i]],writeback_cast=True)
                yield i,fused,live[i],True
                yield i+1,dict(self.g.events[i+1],writeback_cast_join=True),live[i+1],True
                i+=2
                continue
            if i in self.column_epilogues and (i+1>=end or i in boundaries or i+1 in boundaries
                    or not np.array_equal(live[i],live[i+1])):
                # A partial phase with other live demands must retain the
                # ordinary intermediate rather than fuse only some of it.
                del self.column_epilogues[i]
            if i in self.column_epilogues and np.array_equal(live[i],live[i+1]):
                # The original intermediate has exactly one consumer. Each
                # strip preserves its native cast before GELU in the same
                # existing scratch owner, then stores the real GELU version.
                source=event['inputs'][0];target=self.column_epilogues[i]['outputs'][0]
                row_bytes=source['shape'][-1]*source['element_bytes']+target['shape'][-1]*target['element_bytes']
                group=max(self.c['array_rows'],self.c['feature_sram_bytes']//row_bytes//self.c['array_rows']*self.c['array_rows'])
                group=min(group,self.c['queue_entries']*self.c['array_rows'])
                selected=self.row_order(i,live[i])
                if self.c.get('row_chain_overlap',False):self.chain_floor=self.cycle
                for begin in range(0,len(selected),group):
                    self.chain_end=begin+group>=len(selected)
                    yield i,event,selected[begin:begin+group],self.chain_end
                if not len(selected):yield i,event,selected,True
                self.stats['column_gelu_fused_events']+=1
                i+=2
                continue
            from .feeder import use_dense_outer
            if use_dense_outer(self.c,event) and not self.c.get('dense_outer_row_chain',False):
                # Completing the native M domain keeps one weight stripe
                # resident. Its downstream epilogue waits for real stores;
                # no full-width intermediate receives a free scratch owner.
                yield i,event,live[i],True
                i+=1
                continue
            sparse_start=self.c.get('stream_sparse_epilogues',False) and event['kind'] in ('spconv','ME')
            if (independent(event) or sparse_start) and rows(event['outputs'][0])>self.c['array_rows']:
                weight_bytes=self.chain_weight_bytes(event)
                matrices=int(event['kind'] in ('linear','spconv','ME') or
                             (event['kind']=='ancestor_merge' and event['inline_fine']))
                matrix_limit=self.c.get('row_chain_max_matrices',0)
                producers={key(s):s for s in event['outputs']}
                while chain[-1]+1<end and chain[-1] not in boundaries:
                    j=chain[-1]+1;following=self.g.events[j]
                    if j in self.writeback_casts:break
                    next_weights=self.chain_weight_bytes(following)
                    if not independent(following) or rows(following['outputs'][0])!=rows(event['outputs'][0]):break
                    next_matrix=int(following['kind']=='linear' or
                                    (following['kind']=='ancestor_merge' and following['inline_fine']))
                    if matrix_limit and matrices+next_matrix>matrix_limit:break
                    if not across_metadata and following['stage']!=event['stage']:break
                    if not matching_rows(following,producers):break
                    if next_weights and weight_bytes+next_weights>self.c['weight_sram_bytes']:break
                    chain.append(j);weight_bytes+=next_weights
                    matrices+=next_matrix
                    producers.update({key(s):s for s in following['outputs']})
            if len(chain)==1:
                yield i,event,live[i],True
            else:
                gather=chain[-1]+1
                plan=(self.final_gather_plan(gather,self.g.events[chain[-1]]['outputs'][0])
                      if gather<end and len(live[gather]) and chain[-1] not in boundaries else None)
                if (plan is not None and plan['consumer_ids'] is not None and
                    not np.isin(live[gather],plan['consumer_ids']).all()):
                    raise ValueError('Gather demand exceeds explicit dispatched consumers')
                union=np.unique(np.concatenate([live[j] for j in chain]))
                if plan is not None:
                    suppliers=self.g.inputs_for(gather,live[gather])[0][1]
                    union=np.union1d(union,suppliers)
                if not len(union):
                    # A previous phase may have produced every supplier. The
                    # control observation at a checkpoint still must execute.
                    for j in chain:yield j,self.g.events[j],live[j],True
                    i=chain[-1]+1
                    continue
                union=self.row_order(chain[-1],union)
                width=max(self.g.events[j]['outputs'][0]['shape'][-1]*self.g.events[j]['outputs'][0]['element_bytes'] for j in chain)
                output_budget=self.output_budget()
                group=max(self.c['array_rows'],output_budget//width//self.c['array_rows']*self.c['array_rows'])
                oversized=any(s['bytes']>self.c['weight_sram_bytes'] for j in chain
                              for s in self.g.events[j]['inputs'] if s['role']=='weight')
                if self.c.get('dense_output_columns',0) and (not self.c.get('dense_columns_oversized_only',False) or oversized):
                    # Intermediate rows use the existing feature cache, not a
                    # full-width output scratch allocation. Account spills by
                    # ordinary cache accesses; this only changes chain order.
                    group=max(self.c['array_rows'],min(self.c['queue_entries']*self.c['array_rows'],
                        self.c['feature_sram_bytes']//(2*width)//self.c['array_rows']*self.c['array_rows']))
                from .feeder import consumer_resident_rows
                group=consumer_resident_rows(self.c,[self.g.events[j] for j in chain],group)
                active={j:set(live[j].tolist()) for j in chain}
                gather_live=np.zeros(rows(self.g.events[gather]['outputs'][0]),bool) if plan is not None else None
                if plan is not None:gather_live[live[gather]]=True
                filtered=plan is not None and not gather_live.all()
                if plan is not None and plan['exact_dispatch']:
                    # This microprogram consumes its entire committed request
                    # stream exactly once. The list already IS the demand;
                    # a second global-row bitmap would repeat that decision.
                    # This is a protocol assertion, never a future-live-row
                    # predicate that chooses this execution mode at run time.
                    if not np.array_equal(np.sort(live[gather]),np.sort(plan['consumer_ids'])):
                        raise ValueError('Complete dispatch cannot filter or defer request tokens')
                    filtered=False
                    # Decode the complete-list opcode and register its mode
                    # through the existing control issue path, before rows
                    # enter the chain. No new descriptor or SRAM is added.
                    self.memory.clock=max(self.memory.clock,self.cycle)+self.c['issue_cycles']
                    self.cycle=max(self.cycle,self.memory.clock)
                    self.stats['gather_exact_dispatch_control_cycles']+=self.c['issue_cycles']
                    self.stats['gather_exact_dispatch_rows']+=len(plan['consumer_ids'])
                if filtered and not plan['bitmap_reserved']:
                    bitmap_bytes=math.ceil(len(gather_live)/32)*4
                    if bitmap_bytes>=self.control_capacity():raise ValueError('No control SRAM for final-demand bitmap')
                    self.c=dict(self.c,control_sram_bytes=self.c['control_sram_bytes']-bitmap_bytes)
                    self.stats['gather_demand_bitmap_bytes']+=bitmap_bytes
                    plan['bitmap_reserved']=True
                if filtered:self.initialize_gather_demand(live[gather],len(gather_live))
                if self.c.get('row_chain_overlap',False):self.chain_floor=self.cycle
                for begin in range(0,len(union),group):
                    self.chain_window_ordinal=begin//group
                    block=union[begin:begin+group]
                    for j in chain:
                        selected=np.array([row for row in block if row in active[j]],dtype=np.int64)
                        self.chain_end=begin+group>=len(union) and j==chain[-1] and plan is None
                        yield j,self.g.events[j],selected,begin+group>=len(union)
                    if plan is not None:
                        self.prepare_final_gather(gather,plan)
                        offsets=plan['offsets'];order=plan['order']
                        lists=[np.arange(offsets[row],offsets[row+1]) for row in block]
                        links=np.concatenate(lists) if lists else np.empty(0,np.int64)
                        selected=order[links]
                        # Consumer IDs are available only after the real CSR
                        # reads; they then address the finite demand banks.
                        address=np.r_[plan['base']+block*4,plan['base']+(block+1)*4,
                            plan['base']+len(offsets)*4+links*4]
                        ready=self.memory.access(np.unique(address//SECTOR_BYTES))
                        self.cycle=max(self.cycle,ready);self.memory.clock=max(self.memory.clock,self.cycle)
                        if filtered:
                            words=np.unique(selected//32)
                            banks=words%self.c['cam_lookups_per_cycle']
                            cost=int(np.bincount(banks,minlength=self.c['cam_lookups_per_cycle']).max(initial=0))+self.c['cam_latency_cycles']
                            cost+=math.ceil(len(selected)/self.c['vector_lanes'])
                            self.cycle=self.issue_vector(max(self.cycle,self.vector,self.memory.clock),cost)
                            self.memory.clock=self.cycle
                            self.stats['gather_demand_control_read_bytes']+=len(words)*4
                            self.stats['vector_cycles']+=math.ceil(len(selected)/self.c['vector_lanes'])
                            selected=selected[gather_live[selected]]
                        self.chain_end=begin+group>=len(union)
                        yield gather,self.g.events[gather],selected,begin+group>=len(union)
                        if self.allow_phase_retirement(control) and not self.protected_storage(plan['source']):
                            sectors=self.g.sectors(plan['source'],block)
                            tags=len(np.unique(sectors//SLOT_SECTORS))
                            self.memory.clock+=math.ceil(tags/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
                            self.cycle=max(self.cycle,self.memory.clock)
                            self.stats['dead_intermediate_bytes']+=self.memory.discard(sectors,self.cycle)
                        self.stats['streamed_gather_windows']+=1
                self.stats['fused_chains']+=1
                if plan is not None:chain.append(gather)
            i=chain[-1]+1

    def store_sectors(self,sectors,masks,ready):
        """Preserve other live bytes when a partial sector must write-allocate.

        The directory tracks 32B valid/dirty sectors. Read/merge before masked
        stores rather than treating an absent destination's other bytes as
        valid. Resident destinations hit normal SRAM; evicted ones reload.
        """
        partial=sectors[masks!=np.uint64((1<<SECTOR_BYTES)-1)]
        if len(partial):
            ready=max(ready,self.memory.access(partial))
            merge=math.ceil(len(partial)*SECTOR_BYTES/self.c['vector_lanes'])
            ready=self.issue_vector(ready,merge)
            self.stats['vector_cycles']+=merge
            self.stats['partial_store_sectors']+=len(partial)
            self.stats['partial_store_merge_read_bytes']+=len(partial)*SECTOR_BYTES
        return self.memory.access(sectors,write=True,ready=ready)

    def can_drain_output(self,spec,count,matrix=False):
        root=self.g.events[-1]['outputs'][0]
        return (self.c.get('stream_final_output',False) and not matrix and
            key(spec)==key(root) and key(spec) not in self.last_use and
            spec['contiguous'] and spec['address']==root['address'] and
            spec['shape']==root['shape'] and len(spec['shape'])==2 and
            (self.g.bases[key(spec)]+spec['address']-self.g.storage[spec['storage']]['address'])%DMA_BURST_BYTES==0 and
            spec['shape'][-1]*spec['element_bytes']%DMA_BURST_BYTES==0 and
            count*spec['shape'][-1]*spec['element_bytes']<=self.output_budget())

    def output_budget(self):
        if self.attention_output_layout_active:return self.c['output_sram_bytes']//self.c['output_buffers']
        if not self.c.get('finite_operand_feeder',False):return self.c['output_sram_bytes']//self.c['output_buffers']
        from .feeder import operand_capacity
        return operand_capacity(self.c,4)[2]

    def store(self,spec,selected,ready):
        sectors,masks=self.g.write_masks(spec,selected)
        return self.store_sectors(sectors,masks,ready)

    def execute(self,live,control=None):
        self.coverage_phase_active=(self.b and not (
            self.a and control is not None and control[3]=='pilot' and
            self.c.get('coverage_bypass_pilot',False)))
        if self.coverage_row_chain_limit is not None:
            self.c['row_chain_max_matrices']=(self.coverage_row_chain_limit
                if self.coverage_phase_active else self.pilot_row_chain_limit)
        if self.coverage_phase_active and self.coverage_preparation_meta is not None:
            self.finish_coverage_preparation()
            self.cycle=max(self.cycle,self.coverage_preparation_ready)
            self.memory.clock=max(self.memory.clock,self.cycle)
        if getattr(self.g,'inplace_feature_copy_versions',None):
            if control is not None or getattr(self,'fixed_work_copy_executed',False):
                raise ValueError('In-place feature slice copies require one full-depth execution without deferred feature phases')
            self.fixed_work_copy_executed=True
        if control is not None:
            self.protected={key(self.g.events[-1]['outputs'][0])}
            self.protected.update(self.phase_suppliers or ())
            diagnostic_end=(None if control[3]=='pilot' else self.g.maps.get('observation_end'))
            self.protected.update(key(spec) for cp in self.g.maps.get('checkpoints',[])[self.g.maps.get('observation_begin',0):diagnostic_end] for spec in cp['readout_inputs'])
        copied_versions=set()
        carry_use={}
        for at,(old,_,_) in getattr(live,'carry',{}).items():
            carry_use[old]=max(carry_use.get(old,-1),at)
        produced={key(s) for e in self.g.events for s in e['outputs']}
        fork_join=self.g.maps.get('diagnostic_fork_join',-1)
        fork_floor=self.cycle
        fork_join_charged=False
        async_begin=self.g.maps.get('diagnostic_async_begin',len(self.g.events))
        async_floor=None
        for index,event,selected,last_chunk in self.schedule(live,control):
            if event['kind']=='sdpa' and self.attention_staging_storage_bytes:
                from .attention import reset_staging
                reset_staging(self)
            checkpoint_active=self.checkpoint_fork.enter(self,index)
            fork_active=index<fork_join
            async_active=index>=async_begin
            if async_active and async_floor is None:async_floor=self.cycle
            if index==fork_join and not fork_join_charged:
                # Release the single controller descriptor at the query join.
                # No unit horizon is used as an input-readiness shortcut.
                cost=math.ceil(DIAGNOSTIC_FORK_DESCRIPTOR_BYTES/(4*self.c['cam_lookups_per_cycle']))+self.c['cam_latency_cycles']
                self.memory.clock=max(self.memory.clock,self.cycle)+cost
                self.stats['diagnostic_fork_control_cycles']+=cost
                self.stats['control_sram_read_bytes']+=DIAGNOSTIC_FORK_DESCRIPTOR_BYTES
                fork_join_charged=True
            inherited=getattr(live,'carry',{}).get(index)
            if inherited is not None and index not in copied_versions:
                copied_versions.add(index)
                old,new,tokens=inherited
                # Logical inheritance can be narrower than the memory sector.
                # The copy datapath uses byte-write strobes, preserving other
                # new-version values already produced in a previous phase.
                offsets,masks=self.g.inheritance_masks(old[0],tokens)
                self.stats['version_copy_logical_bytes']+=len(tokens)*self.g.dependency_granule(old[0])
                self.stats['version_copy_masked_sectors']+=int((masks!=np.uint64((1<<SECTOR_BYTES)-1)).sum())
                group=max(1,self.c['output_sram_bytes']//self.c['output_buffers']//SECTOR_BYTES)
                for begin in range(0,len(offsets),group):
                    block=offsets[begin:begin+group]
                    self.memory.clock=max(self.memory.clock,self.cycle)
                    ready=self.memory.access(block+self.g.bases[old]//SECTOR_BYTES)
                    ready=self.issue_vector(ready,math.ceil(len(block)*SECTOR_BYTES/self.c['vector_lanes']))
                    self.cycle=max(self.cycle,self.store_sectors(block+self.g.bases[new]//SECTOR_BYTES,
                        masks[begin:begin+group],ready))
                    self.stats['version_copy_bytes']+=len(block)*SECTOR_BYTES
            if not len(selected) or not event['outputs'] or (event['outputs'][0]['bytes']==0
                    and event.get('control_primitive') not in ('compact','unique')):
                if index in self.diagnostic_producers:
                    self.diagnostic_completions[index]=max(self.diagnostic_completions.get(index,0),self.cycle)
                if self.chain_end and not fork_active and not checkpoint_active:
                    if not self.c.get('event_local_completion',False):
                        self.cycle=max(self.cycle,self.stats['last_tile_completion'],int(max(self.matrix)),self.vector)
                    self.memory.clock=max(self.memory.clock,self.cycle)
                if last_chunk:self.observe(index,control)
                self.checkpoint_fork.join(self,index,last_chunk)
                continue
            if event.get('writeback_cast_join',False):
                from .writeback import join
                join(self,index,event,last_chunk,control)
                continue
            if event.get('integer_mask_fusion_join',False):
                from .vector_fusion import join
                join(self,index,event,last_chunk,control)
                continue
            if event.get('training_batch_norm_relu_fusion_join',False):
                from .vector_fusion import training_batch_norm_relu_join
                training_batch_norm_relu_join(self,index,event,last_chunk,control)
                continue
            if event.get('training_batch_norm_add_relu_fusion_join',False):
                from .vector_fusion import training_batch_norm_add_relu_join
                training_batch_norm_add_relu_join(self,index,event,last_chunk,control)
                continue
            if index in getattr(self.g,'concat_joins',{}):
                from .concatenation import join
                join(self,index,selected,control,last_chunk)
                self.checkpoint_fork.join(self,index,last_chunk)
                continue
            if index in getattr(self.g,'concat_copies',{}):
                source,destination=self.g.concat_copies[index]
                self.stats['channel_concat_elided_bytes']+=len(selected)*(event['outputs'][0]['shape'][1]-source['shape'][1])*source['element_bytes']
                # Only the unaliased native fine bank still needs a copy.
                # Aliased gather stores already produced the other columns.
                event=dict(event,inputs=[source],outputs=[destination],op='aten.clone.default',concat_copy=True)
            before=self.cycle
            event_completed=self.cycle
            before_horizon=max(before,self.stats['last_tile_completion'],int(max(self.matrix)),self.vector)
            if self.c.get('profile_events',False):
                before_counts=self.memory.counts.copy();before_stats=self.stats.copy()
            if self.c.get('consumer_attention_reclaim_active',False) and event['kind']=='sdpa':
                from .attention import set_output_layout
                set_output_layout(self,True)
            from .normalization_timing import current_statistics,prepare as prepare_normalization,start_stream
            normalization=prepare_normalization(self,index,event) if current_statistics(event) else None
            start_stream(self,index,live)
            if event['kind']=='spconv':
                name=event['map_key'];base=self.g.map_bases[name]
                mapping=np.asarray(self.g.maps['maps'][name]['mapping'])
                if base not in self.prepared_maps:
                    self.prepared_maps.add(base)
                    # Three coordinate additions and one map comparison for
                    # every offset/output candidate; table construction writes
                    # every real captured map slot, including invalid markers.
                    geometry=math.ceil(mapping.size*4/self.c['vector_lanes'])
                    geometry_done=self.issue_vector(self.cycle,geometry)
                    self.memory.clock=max(self.memory.clock,self.cycle)
                    done=self.memory.access(np.arange(base//SECTOR_BYTES,math.ceil((base+mapping.nbytes)/SECTOR_BYTES)),write=True,ready=geometry_done)
                    self.cycle=max(self.cycle,done)
                    self.stats['geometry_cycles']+=geometry
            elif event['kind']=='ME':
                name=event['map_key'];base=self.g.map_bases[name]
                rule=self.g.maps['maps'][name]
                if base not in self.prepared_maps:
                    self.prepared_maps.add(base)
                    nbytes=int(rule['pair_byte_offsets'][-1])
                    geometry=math.ceil(sum(len(pair[0]) for pair in rule['offsets'])*4/self.c['vector_lanes'])
                    geometry_done=self.issue_vector(self.cycle,geometry)
                    self.memory.clock=max(self.memory.clock,self.cycle)
                    done=self.memory.access(np.arange(base//SECTOR_BYTES,math.ceil((base+nbytes)/SECTOR_BYTES)),write=True,ready=geometry_done)
                    self.cycle=max(self.cycle,done)
                    self.stats['geometry_cycles']+=geometry
            ancestor=event['kind']=='ancestor_merge'
            matrix=event['kind'] in ('linear','spconv','ME','sdpa','attention') or event.get('op')=='aten.mm.default' or (ancestor and event['inline_fine'])
            feeder=matrix and event['kind'] not in ('sdpa','attention') and self.c.get('finite_operand_feeder',False)
            sparse_column_feeder=False
            if feeder:
                from .feeder import use_sparse_columns,use_dense_outer,dense_consumer
                sparse_column_feeder=use_sparse_columns(self.c,event,len(selected))
            dense_outer_feeder=feeder and use_dense_outer(self.c,event)
            column_feeder=sparse_column_feeder or (feeder and dense_consumer(self.c,event) and bool(self.c.get('dense_output_columns',0)) and
                (not self.c.get('dense_columns_oversized_only',False) or event['inputs'][1]['bytes']>self.c['weight_sram_bytes']))
            selected=self.row_order(index,selected)
            out=event['outputs'][0]
            from .integer_stream import needed as external_integer_needed
            external_integer=bool(event.get('control_primitive') and
                external_integer_needed(event,self.c,self.output_budget()))
            if external_integer and not np.array_equal(np.sort(selected),np.arange(rows(out))):
                raise ValueError('External integer microprogram requires its complete output domain')
            row_bytes=max(1,(out['shape'][-1] if out['shape'] else 1)*out['element_bytes'])
            buffer_rows=max(1,self.output_budget()//row_bytes)
            if matrix:
                bm=max(self.c['array_rows'],min(buffer_rows//self.c['array_rows'],self.c['queue_entries'])*self.c['array_rows'])
            if feeder:
                from .feeder import operand_capacity
                _,_,accumulator_budget=operand_capacity(self.c,out['element_bytes'])
                if sparse_column_feeder or dense_outer_feeder:
                    # The N-outer helper creates its own finite M windows.
                    # This outer descriptor never owns a full-width scratch.
                    bm=len(selected)
                else:
                    accumulator_width=min(out['shape'][-1],self.c['dense_output_columns']) if column_feeder else out['shape'][-1]
                    bm=accumulator_budget//(accumulator_width*4)
                    if column_feeder:bm=min(self.c['queue_entries']*self.c['array_rows'],bm)
                    if bm>=self.c['array_rows']:bm=bm//self.c['array_rows']*self.c['array_rows']
                    if bm<=0:raise ValueError('No accumulator row fits output SRAM')
            else:
                bm=max(1,buffer_rows)
            from .segments import descriptor as segment_descriptor
            segment=segment_descriptor(self.g,index) is not None
            if segment:
                from .segments import group_rows as segment_group_rows
                bm=min(bm,segment_group_rows(self,index,selected))
            # Native Flash rows are kept together until patch bookkeeping is
            # available; the supported captured traces use explicit SDPA.
            if event['kind']=='attention':bm=len(selected)
            if event['kind']=='attention':
                raise NotImplementedError('Native Flash requires an explicit patch schedule; capture the supported SDPA path')
            if event['kind']=='sdpa':
                # Materialized QK scores remain in this finite output buffer.
                # The schedule below does not assume a free unbounded matrix.
                score_row_bytes=out['shape'][-2]*4+row_bytes
                bm=max(1,min(bm,self.output_budget()//score_row_bytes))
                if self.c.get('finite_operand_feeder',False):
                    # One head/patch query window owns Q/K/V and all scores
                    # until AV completes. Independent query tiles can share
                    # this owner: AV may then issue to several original
                    # arrays instead of exposing one long-K product at a time.
                    from .attention import cohort_rows
                    bm=cohort_rows(self,out['shape'][-2],out['shape'][-1],out['element_bytes'])
                if self.c.get('attention_ready_queue',False):
                    from .ready_queue import ReadyQueue
                    self.attention_queues=[ReadyQueue(self.attention_queue_capacity) for _ in self.matrix]
                    # Unknown holes in previous feeder programs are not free.
                    # Only this attention controller owns these reservations.
                    for queue,available in zip(self.attention_queues,self.matrix):
                        queue.floor=int(available);queue.horizon=int(available)
            for ids in ([selected] if sparse_column_feeder or dense_outer_feeder or external_integer else self.groups(index,selected,bm)):
                if event['kind']=='sdpa' and self.c.get('finite_operand_feeder',False) and len(np.unique(ids//out['shape'][-2]))!=1:
                    raise ValueError('A finite attention window may not span multiple heads/patches')
                # Finite queue: no unbounded hidden buffer of prefetched tasks.
                window_managed=column_feeder and self.c.get('column_window_overlap',False)
                buffer=int(np.argmin(self.buffer_ready))
                if not window_managed:self.active_output_buffer=buffer
                self.memory.clock=max(self.memory.clock,self.cycle,0 if window_managed else int(self.buffer_ready[buffer]))
                from .normalization_timing import references
                refs=([(event['inputs'][0],ids)] if event.get('concat_copy',False)
                      else references(self.g,index,ids,event))
                input_ready=self.memory.clock
                if normalization is not None:
                    input_ready=max(input_ready,self.memory.access(normalization['mean']))
                    input_ready=max(input_ready,self.memory.access(normalization['inverse']))
                if event['kind']=='spconv' and not sparse_column_feeder:
                    name=event['map_key'];mapping=np.asarray(self.g.maps['maps'][name]['mapping'])
                    offsets=(np.arange(mapping.shape[0])[:,None]*mapping.shape[1]+ids)*mapping.itemsize+self.g.map_bases[name]
                    input_ready=max(input_ready,self.memory.access(np.unique(offsets//SECTOR_BYTES)))
                elif event['kind']=='ME' and not sparse_column_feeder:
                    name=event['map_key'];rule=self.g.maps['maps'][name]
                    active=np.zeros(rows(out),dtype=bool);active[ids]=True
                    address=[]
                    for offset,pair in enumerate(rule['offsets']):
                        selected_edges=np.flatnonzero(active[pair[1]])
                        for axis,values in enumerate(pair):
                            start=self.g.map_bases[name]+int(rule['pair_byte_offsets'][offset])
                            if axis:start+=pair[0].nbytes
                            address.append((start+selected_edges*values.itemsize)//SECTOR_BYTES)
                    input_ready=max(input_ready,self.memory.access(np.unique(np.concatenate(address))))
                logical_reads=0
                ordinary_refs=([] if feeder or external_integer or event.get('diagnostic_source_fused',False) else refs)
                indexed_here=event.get('indexed_source',False) and not (column_feeder or dense_outer_feeder)
                if feeder and indexed_here:ordinary_refs=[(event['inputs'][-1],ids)]
                if ancestor and feeder:
                    ordinary_refs=[(s,r) for s,r in refs if s is not event['inputs'][0] and s is not event['inputs'][22]]
                if event['kind'] in ('progressive_difference','progressive_update',
                                     'progressive_observe'):
                    # Global state and parent addresses are not available until
                    # the real row selector (and then parent map) has returned.
                    selector_position={'progressive_difference':3,'progressive_update':2}.get(event['kind'],1)
                    selector=event['inputs'][selector_position]
                    stages=[next((s,r) for s,r in ordinary_refs if s is selector)]
                    if event['kind']!='progressive_update':
                        mapping=event['inputs'][2]
                        stages.append(next((s,r) for s,r in ordinary_refs if s is mapping))
                    for spec,source_rows in stages:
                        sectors=self.g.sectors(spec,source_rows)
                        input_ready=max(input_ready,self.memory.access(sectors))
                        logical_reads+=len(sectors)*self.memory.sector_bytes
                        cost=math.ceil(len(ids)/self.c['vector_lanes'])
                        input_ready=self.issue_vector(input_ready,cost)
                        self.stats['vector_cycles']+=cost
                        self.stats['progressive_address_cycles']+=cost
                        self.memory.clock=max(self.memory.clock,input_ready)
                    ordinary_refs=[(s,r) for s,r in ordinary_refs if not any(s is t for t,_ in stages)]
                if (segment or event.get('indexed_source',False) or event['kind']=='progressive_center_distance' or 'diagnostic_control_rows' in event or (event.get('op','').startswith(('aten.index.Tensor','aten.index_select','aten.gather'))
                    and self.g.maps['indices'].get(index))):
                    indices=[(s,r) for s,r in ordinary_refs if s['role']=='index']
                    for spec,source_rows in indices:
                        sectors=self.g.sectors(spec,source_rows)
                        input_ready=max(input_ready,self.memory.access(sectors))
                        logical_reads+=len(sectors)*self.memory.sector_bytes
                    if segment:
                        from .segments import decode as segment_decode
                        input_ready=segment_decode(self,ids,input_ready)
                    if event['kind']=='progressive_center_distance' or indexed_here:
                        # The captured parent values describe actual accesses;
                        # their addresses are usable only after map return.
                        cost=math.ceil(len(ids)/self.c['vector_lanes'])
                        input_ready=self.issue_vector(input_ready,cost)
                        self.stats['vector_cycles']+=cost
                        label='indexed_affine' if event.get('indexed_source',False) else 'center_distance'
                        self.stats[label+'_address_cycles']+=cost
                    self.memory.clock=max(self.memory.clock,input_ready)
                    ordinary_refs=[(s,r) for s,r in ordinary_refs if s['role']!='index']
                reuse_attention_kv=False
                if event['kind']=='sdpa' and self.attention_staging_storage_bytes:
                    from .attention import reuse_staging
                    reuse_attention_kv=reuse_staging(self,index,ids)
                    input_ready=max(input_ready,self.memory.clock)
                    if reuse_attention_kv:
                        # SDPA references are Q,K,V in operand order. Q can
                        # alias the very same spec object as K or V; removing
                        # references by object identity would lose its load.
                        self.stats['attention_coverage_staging_saved_source_bytes']+=sum(
                            len(self.g.sectors(s,r))*SECTOR_BYTES for s,r in ordinary_refs[1:3])
                        ordinary_refs=ordinary_refs[:1]+ordinary_refs[3:]
                for spec,source_rows in ordinary_refs:
                    sectors=self.g.sectors(spec,source_rows)
                    input_ready=max(input_ready,self.memory.access(sectors,weight=spec['role']=='weight'))
                    logical_reads+=len(sectors)*self.memory.sector_bytes
                flops=self.g.flops(index,ids)
                if sparse_column_feeder:
                    from .feeder import sparse_columns
                    done=sparse_columns(self,index,ids)
                elif dense_outer_feeder:
                    from .feeder import dense_outer_rows
                    done=dense_outer_rows(self,index,ids)
                elif column_feeder:
                    from .feeder import dense_columns
                    self.memory.clock=max(self.memory.clock,input_ready)
                    done=dense_columns(self,index,ids)
                elif feeder:
                    from .feeder import execute as feed_products
                    # The captured rulebook gives topology, not a free early
                    # address: feature requests wait for its actual return.
                    self.memory.clock=max(self.memory.clock,input_ready)
                    done=feed_products(self,index,ids)
                    # Stream the resident FP32 accumulator through the vector
                    # output path. Bias/cast fuse into this final transfer.
                    done=self.output_transfer(len(ids)*out['shape'][-1]*4,done,offset=self.accumulator_base(out['element_bytes']))
                    # Bias is a final pointwise update of the resident FP32
                    # accumulator, read only after the products are ready.
                    if len(event['inputs'])>2+int(event.get('indexed_source',False)) and not ancestor:
                        bias=event['inputs'][2]
                        if bias['bytes']:
                            ready=self.memory.access(self.g.sectors(bias,np.arange(rows(bias))),weight=bias['role']=='weight')
                            cost=math.ceil(len(ids)*out['shape'][-1]/self.c['vector_lanes'])
                            done=self.issue_vector(max(done,ready),cost)
                            self.stats['vector_cycles']+=cost
                    if out['element_bytes']<4:
                        cost=math.ceil(len(ids)*out['shape'][-1]/self.c['vector_lanes'])
                        done=self.issue_vector(done,cost)
                        self.stats['vector_cycles']+=cost
                    if ancestor:
                        cost=self.vector_cost(event,len(ids))
                        done=self.issue_vector(done,cost)
                        self.stats['vector_cycles']+=cost
                elif matrix:
                    if event['kind']=='sdpa' and self.c.get('consumer_attention_pipeline_active',False):
                        from .attention import pipeline
                        done=pipeline(self,len(ids),out['shape'][-2],out['shape'][-1],
                                      out['element_bytes'],input_ready,reuse_kv=reuse_attention_kv)
                        if self.attention_staging_storage_bytes:
                            from .attention import remember_staging
                            remember_staging(self,index,ids,done)
                        self.stats['matrix_flops']+=flops
                    else:
                        width=out['shape'][-1]
                        ntiles=math.ceil(width/self.c['array_cols'])
                        precision=self.c['fp32_mac_cycles'] if out['element_bytes']==4 else 1
                        k=event['inputs'][0]['shape'][-1]
                        if event['kind']=='spconv':
                            mapping=np.asarray(self.g.maps['maps'][event['map_key']]['mapping'])
                            edge_counts=(mapping[:,ids]>=0).sum(1)
                            rowtiles=int(((edge_counts+self.c['array_rows']-1)//self.c['array_rows']).sum())
                        elif event['kind']=='ME':
                            active=np.zeros(rows(out),dtype=bool);active[ids]=True
                            edge_counts=[active[np.asarray(pair[1])].sum() for pair in self.g.maps['maps'][event['map_key']]['offsets']]
                            rowtiles=sum(math.ceil(n/self.c['array_rows']) for n in edge_counts)
                        else:rowtiles=math.ceil(len(ids)/self.c['array_rows'])
                        tile_cycles=(k*precision+self.c['array_rows']+self.c['array_cols']-2)
                        done=input_ready
                        tile_count=ntiles*rowtiles
                        av_count=0;av_duration=0
                        if event['kind']=='sdpa':
                            sequence=out['shape'][-2];dim=out['shape'][-1]
                            groups,counts=np.unique(ids//sequence,return_counts=True)
                            qt=sum(math.ceil(n/self.c['array_rows']) for n in counts)
                            tile_cycles=dim*precision+self.c['array_rows']+self.c['array_cols']-2
                            tile_count=qt*math.ceil(sequence/self.c['array_cols'])
                            av_duration=sequence*precision+self.c['array_rows']+self.c['array_cols']-2
                            av_count=qt*math.ceil(dim/self.c['array_cols'])
                            if self.c.get('finite_operand_feeder',False):
                                element_bytes=out['element_bytes']
                                payload=(len(ids)+2*sequence)*dim*element_bytes
                                score_bytes=len(ids)*sequence*4
                                av_bytes=len(ids)*dim*4
                                staging_bytes=payload+score_bytes+av_bytes
                                if staging_bytes>self.output_budget():
                                    raise AssertionError('Attention output slot capacity exceeded')
                                self.stats['attention_peak_staging_bytes']=max(self.stats['attention_peak_staging_bytes'],staging_bytes)
                                scratch=self.accumulator_base(4)
                                qbytes=len(ids)*dim*element_bytes;kvbytes=sequence*dim*element_bytes
                                score_base=scratch+payload;av_base=score_base+score_bytes
                                input_ready=self.output_transfer(payload,input_ready,write=True,offset=scratch)
                                # Each Q is consumed for every K tile. Every
                                # independent M tile consumes K/V again from the
                                # same bounded staging owner; no free multicast.
                                for _ in range(math.ceil(sequence/self.c['array_cols'])):
                                    input_ready=self.output_transfer(qbytes,input_ready,offset=scratch)
                                for _ in range(qt):
                                    input_ready=self.output_transfer(kvbytes,input_ready,offset=scratch+qbytes)
                        initiation=tile_cycles-(self.c['array_rows']+self.c['array_cols']-2) if self.c['array_wave_pipeline'] else 0
                        done=self.issue_matrix(tile_count,tile_cycles,input_ready,initiation)
                        self.stats['array_occupied_cycles']+=tile_count*(initiation or tile_cycles)
                        if event['kind'] in ('sdpa','attention'):
                            softmax=math.ceil(len(ids)*out['shape'][-2]*SOFTMAX_PASSES/self.c['vector_lanes'])
                            # QK must complete before normalization; AV cannot
                            # consume attention probabilities before softmax.
                            if self.c.get('finite_operand_feeder',False):
                                done=self.output_transfer(score_bytes,done,write=True,offset=score_base)
                                probabilities=len(ids)*sequence*element_bytes
                                from .attention import normalize,use_registers
                                if use_registers(self.c,sequence):
                                    done=normalize(self,len(ids),sequence,element_bytes,score_base,done)
                                    self.stats['attention_scratch_bytes']+=score_bytes*2+probabilities*2
                                else:
                                    # Original materialized max/exp/sum/normalize.
                                    for _ in range(3):done=self.output_transfer(score_bytes,done,offset=score_base)
                                    done=self.output_transfer(score_bytes,done,write=True,offset=score_base)
                                    done=self.issue_vector(done,softmax)
                                    self.stats['vector_cycles']+=softmax
                                    done=self.output_transfer(probabilities,done,write=True,offset=score_base)
                                    self.stats['attention_scratch_bytes']+=score_bytes*5+probabilities*2
                                done=self.output_transfer(probabilities,done,offset=score_base)
                                for _ in range(qt):
                                    done=self.output_transfer(kvbytes,done,offset=scratch+qbytes+kvbytes)
                            else:
                                scratch=math.ceil(len(ids)*out['shape'][-2]*4*3/(self.c['sram_banks']*self.c['bank_bytes_per_cycle']))
                                done=self.issue_vector(done,softmax+scratch)
                                self.stats['vector_cycles']+=softmax+scratch
                                self.stats['attention_scratch_bytes']+=len(ids)*out['shape'][-2]*4*3
                            av_initiation=sequence*precision if self.c['array_wave_pipeline'] else 0
                            done=self.issue_matrix(av_count,av_duration,done,av_initiation)
                            self.stats['array_occupied_cycles']+=av_count*(av_initiation or av_duration)
                            if self.c.get('finite_operand_feeder',False):
                                done=self.output_transfer(av_bytes,done,write=True,offset=av_base)
                                done=self.output_transfer(av_bytes,done,offset=av_base)
                                cast_cycles=math.ceil(len(ids)*dim/self.c['vector_lanes'])
                                done=self.issue_vector(done,cast_cycles)
                                self.stats['vector_cycles']+=cast_cycles
                        self.stats['matrix_flops']+=flops
                elif event.get('control_primitive'):
                    if external_integer:
                        from .integer_stream import execute as execute_integer_control
                    else:
                        from .control_timing import execute as execute_integer_control
                    done=execute_integer_control(self,index,input_ready)
                elif event['kind']=='progressive_sort':
                    from .feeder import consumer_sort
                    done=consumer_sort(self,event,input_ready)
                elif segment:
                    from .segments import reduce as segment_reduce
                    done=segment_reduce(self,index,ids,input_ready)
                elif 'diagnostic_unique_native' in event:
                    from .diagnostic_pipeline import unique_index
                    done=unique_index(self,event,ids,input_ready)
                elif event.get('diagnostic_source_fused',False):
                    if self.c.get('diagnostic_register_source',False):
                        from .diagnostic_pipeline import register_source
                        done=register_source(self,event,ids,input_ready)
                    else:
                        done=self.materialized_diagnostic_source(event,ids,input_ready)
                else:
                    cost=self.vector_cost(event,len(ids))
                    done=self.issue_vector(input_ready,cost)
                    self.stats['vector_cycles']+=cost
                    if event.get('diagnostic_control',False):
                        self.stats['diagnostic_control_vector_cycles']+=cost
                if event.get('integer_mask_fusion',False):
                    self.stats['integer_mask_fused_elements']+=len(ids)*out['shape'][-1]
                    self.stats['integer_mask_elided_bytes']+=(
                        len(ids)*event['integer_mask_elided_bytes_per_row'])
                if event.get('training_batch_norm_relu_fusion',False):
                    cost=math.ceil(len(ids)*out['shape'][-1]/self.c['vector_lanes'])
                    done=self.issue_vector(done,cost)
                    self.stats['vector_cycles']+=cost
                    self.stats['training_batch_norm_relu_vector_cycles']+=cost
                    self.stats['training_batch_norm_relu_elements']+=len(ids)*out['shape'][-1]
                    self.stats['training_batch_norm_relu_elided_bytes']+=(
                        2*len(ids)*out['shape'][-1]*out['element_bytes'])
                if event.get('training_batch_norm_add_relu_fusion',False):
                    cost=math.ceil(len(ids)*out['shape'][-1]/self.c['vector_lanes'])
                    done=self.issue_vector(done,cost)
                    self.stats['vector_cycles']+=cost
                    self.stats['training_batch_norm_add_relu_vector_cycles']+=2*cost
                    self.stats['training_batch_norm_add_relu_elements']+=len(ids)*out['shape'][-1]
                    self.stats['training_batch_norm_add_relu_elided_bytes']+=(
                        4*len(ids)*out['shape'][-1]*out['element_bytes'])
                if event.get('writeback_cast',False):
                    cost=math.ceil(len(ids)*out['shape'][-1]/self.c['vector_lanes'])
                    done=self.issue_vector(done,cost)
                    self.stats['vector_cycles']+=cost
                    self.stats['writeback_cast_vector_cycles']+=cost
                    self.stats['writeback_cast_elements']+=len(ids)*out['shape'][-1]
                    self.stats['writeback_cast_elided_bytes']+=len(ids)*self.g.events[index]['outputs'][0]['shape'][-1]*self.g.events[index]['outputs'][0]['element_bytes']
                output_done=done
                for output_index,spec in enumerate(event['outputs']):
                    if not spec['bytes']:continue
                    if external_integer:continue  # Stream already stored every finite output packet.
                    if column_feeder or (event.get('diagnostic_source_fused',False) and self.c.get('diagnostic_register_source',False)):continue
                    written=ids if rows(spec)==rows(out) else np.arange(rows(spec))
                    if event['kind'] in ('progressive_update','indexed_store'):
                        written=np.asarray(self.g.maps['indices'][index][0])[ids] if output_index==0 else ids
                    elif event['kind']=='progressive_observe':
                        written=np.asarray(self.g.maps['indices'][index][0])[ids] if output_index>=2 else ids
                    direct=self.can_drain_output(spec,len(written),matrix)
                    if direct:
                        from .feeder import drain_output
                        nbytes=len(written)*spec['shape'][-1]*spec['element_bytes']
                        m=self.memory;banks,calendar,lengths=m.ports(2)
                        offset=self.accumulator_base(4)
                        output_done=max(output_done,int(drain_output(offset,nbytes,done,m.state,m.pending,
                            banks,calendar,lengths,self.c['bank_bytes_per_cycle'],m.counts,
                            m.burst_cycles,self.c['dram_latency_cycles'],m.dma_activity)))
                        self.stats['direct_output_bytes']+=nbytes
                        self.stats['output_window_write_bytes']+=nbytes
                        self.stats['output_window_read_bytes']+=nbytes
                    else:output_done=max(output_done,self.store(spec,written,done))
                self.stats['issued_tiles']+=1
                self.stats['tile_row_slots']+=bm
                self.stats['useful_rows']+=len(ids)
                self.memory.clock+=self.c['issue_cycles']
                # All tile completion events contribute to the operation's
                # barrier; no next operator consumes an unavailable output.
                self.stats['last_tile_completion']=max(self.stats['last_tile_completion'],output_done)
                event_completed=max(event_completed,output_done)
                if not window_managed:self.buffer_ready[buffer]=output_done
                from .normalization_timing import generated
                generated(self,index,ids,output_done)
                if self.allow_phase_retirement(control):
                    for spec,source_rows in refs:
                        if (spec['role']!='feature' or self.protected_storage(spec)
                            or any(self.same_allocation(out,spec) for out in event['outputs'])
                            or self.last_use.get(key(spec))!=index or carry_use.get(key(spec),-1)>index):continue
                        # A sparse/gather operand can recur in a later tile of
                        # this same last-use operator. Only injective rowwise
                        # consumers permit immediate reclamation here.
                        if not self.can_reclaim(event,spec,self.same_allocation) or not len(source_rows):continue
                        sorted_rows=np.sort(source_rows)
                        cuts=np.flatnonzero(np.diff(sorted_rows)!=1)+1
                        for run in np.split(sorted_rows,cuts):
                            begin_offset=int(self.g.starts(spec,run[:1])[0])
                            nbytes=len(run)*(spec['shape'][-1] if spec['shape'] else 1)*spec['element_bytes']
                            # Reclaim only whole valid granules, never bytes
                            # belonging to an unconsumed neighboring row.
                            begin_token=math.ceil(begin_offset/SECTOR_BYTES)
                            end_token=(begin_offset+nbytes)//SECTOR_BYTES
                            tokens=np.arange(begin_token,end_token,dtype=np.int64)+self.g.bases[key(spec)]//SECTOR_BYTES
                            self.stats['dead_intermediate_bytes']+=self.memory.discard(tokens,done)
            if self.attention_queues is not None:
                self.stats['attention_queue_backpressure_cycles']+=sum(q.backpressure for q in self.attention_queues)
                self.stats['attention_queue_filled_gap_cycles_total']+=sum(q.hidden for q in self.attention_queues)
                self.stats['attention_ready_queue_bytes']=self.attention_queue_storage_bytes
                # The legacy array horizons retain all booked work, including
                # commands not yet issued at the frontend's current time.
                self.attention_queues=None
            completed=(max(self.cycle,event_completed) if self.c.get('event_local_completion',False)
                       else max(self.cycle,self.stats['last_tile_completion'],int(max(self.matrix)),self.vector))
            overlapping=(self.chain_floor is not None and not self.chain_end) or fork_active or async_active or checkpoint_active
            # An independent row window need not wait for every unit's future
            # completion. Actual input slots, feeder owners, bank calendars
            # and output buffers still gate its reads and writes. A full
            # chain/observation boundary drains the complete horizon.
            self.cycle=(self.checkpoint_fork.floor if checkpoint_active else async_floor if async_active else self.chain_floor if self.chain_floor is not None
                        else fork_floor) if overlapping else completed
            if not overlapping:self.memory.clock=max(self.memory.clock,self.cycle)
            if (last_chunk and self.c.get('retire_complete_operands',False)
                and self.allow_phase_retirement(control)):
                for spec in event['inputs']:
                    identity=key(spec)
                    if (spec['role']!='feature' or identity not in produced or self.protected_storage(spec)
                        or self.last_use.get(identity)!=index or carry_use.get(identity,-1)>index
                        or any(self.same_allocation(out,spec) for out in event['outputs'])):continue
                    # Sparse/gather operands recur within this operator. Only
                    # its final chunk retires the whole version, after all
                    # readers finish; unlike per-tile rowwise reclamation.
                    sectors=self.g.sectors(spec,np.arange(rows(spec)))
                    tags=len(np.unique(sectors//SLOT_SECTORS))
                    cost=math.ceil(tags/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
                    self.memory.clock=max(self.memory.clock,self.cycle)+cost
                    if not overlapping:self.cycle=max(self.cycle,self.memory.clock)
                    retirement_horizon=event_completed if fork_active or async_active or checkpoint_active else completed
                    self.stats['retired_operand_bytes']+=self.memory.discard(sectors,max(retirement_horizon,self.memory.clock))
                    self.stats['retirement_control_cycles']+=cost
            if self.c.get('consumer_attention_reclaim_active',False) and self.attention_output_layout_active:
                set_output_layout(self,False)
            self.checkpoint_fork.join(self,index,last_chunk)
            if not self.coverage_phase_active:
                self.service_coverage_preparation()
            elapsed=max(0,max(completed,self.cycle,self.memory.clock)-before_horizon)
            self.stages[event['stage']]+=elapsed
            if self.c.get('profile_events',False):
                self.event_profile.append(dict(index=index,stage=event['stage'],kind=event['kind'],
                    op=event.get('op',''),rows=len(selected),shape=out['shape'],
                    cycles=elapsed,memory=(self.memory.counts-before_counts).tolist(),
                    statistics=dict(self.stats-before_stats)))
            if index in self.diagnostic_producers:
                self.diagnostic_completions[index]=max(self.diagnostic_completions.get(index,0),event_completed)
            if last_chunk:self.observe(index,control)
        if control is not None and control[3]=='pilot':
            self.pilot_main_completed_cycle=self.cycle
        if self.checkpoint_fork.current is not None:raise ValueError('Unfinished checkpoint descriptor')
        return self.cycle

    def observe(self,index,control):
        barrier=self.g.maps.get('diagnostic_control_barriers',{}).get(index)
        if barrier is not None:
            # Score/TV output must exist before the controller can address
            # its response and demand tables. The action must be committed
            # before a following early-output program may consume it.
            producers=barrier.get('producers',[])
            if any(at not in self.diagnostic_completions for at in producers):
                raise ValueError('A diagnostic action reads an unfinished producer')
            ready=max((self.diagnostic_completions[at] for at in producers),default=0)
            self.cycle=max(self.cycle,self.memory.clock,ready)+barrier['cycles']
            self.memory.clock=max(self.memory.clock,self.cycle)
            self.stats['diagnostic_control_port_cycles']+=barrier['cycles']
            self.stats['control_sram_read_bytes']+=barrier['read_bytes']
        if control is None:return
        observations,stops,witnesses,phase=control
        if (phase=='pilot' and uses_witness_selector(self.c)
            and index==self.g.maps['response_context']['after_event']):
            from .diagnostic_pipeline import select_witnesses
            select_witnesses(self,witnesses)
        depths=[d for d,cp in enumerate(self.g.maps['checkpoints'][:-1]) if cp['after_event']==index]
        if depths:self.diagnostics(observations,stops,witnesses,depths=depths,phase=phase)

    def diagnostics(self,observations,stops,witnesses,depths=None,phase=None):
        if self.c.get('finite_diagnostic_pipeline',False):
            from .diagnostic_pipeline import execute
            return execute(self,observations,stops,witnesses,depths,phase)
        total=0;flops=0;vector_ops=0;bytes_=0;emitted=[]
        n=len(stops);classes=observations['scores'].shape[-1]
        for depth,cp in enumerate(self.g.maps['checkpoints'][:-1]):
            if depth<self.g.maps.get('observation_begin',0):continue
            if phase!='pilot' and depth>=self.g.maps.get('observation_end',len(self.g.maps['checkpoints'])-1):continue
            if depths is not None and depth not in depths:continue
            active=stops>=depth
            if phase=='pilot':active&=witnesses
            elif phase=='remaining':active&=~witnesses
            count=int(active.sum())
            if not count:continue
            emitted.extend(np.flatnonzero(active&(stops==depth)).tolist())
            selected=np.flatnonzero(active)
            operands=list(zip(cp['readout_inputs'],cp['readout_maps']))
            projection_count,checks=equivalent_count(operands,selected)
            history=[]
            for prior in self.g.maps['checkpoints'][max(self.g.maps.get('observation_begin',0),depth-self.c['predictor_window']+1):depth+1]:
                history.extend(zip(prior['readout_inputs'],prior['readout_maps']))
            context=self.g.maps.get('response_context')
            if context is not None:history.append((context['spec'],context['mapping']))
            control_count,history_checks=equivalent_count(history,selected)
            # Hierarchy validation and multicast read the existing index maps.
            # Charge the finite vector/lookup path and metadata transfer.
            checks+=history_checks
            total+=math.ceil(checks/self.c['vector_lanes'])
            bytes_+=checks*4
            self.stats['diagnostic_leaf_observations']+=count
            self.stats['diagnostic_projected_rows']+=projection_count
            self.stats['diagnostic_control_rows']+=control_count
            k,width=cp['readout_shape'];lang=cp['semantic_width']
            query_mode=self.c.get('trajectory_space','language')=='query_scores'
            if query_mode:
                # Only a stopped row materializes the wide semantic vector.
                # normalize(readout @ text.T) gives the same query trajectory
                # as normalize(normalize(readout) @ text.T).
                exit_rows=np.flatnonzero(active&(stops==depth))
                output_count,exit_checks=equivalent_count(operands,exit_rows)
                bytes_+=exit_checks*4
                total+=math.ceil(exit_checks/self.c['vector_lanes'])
                matrices=[]
                query_precision=1 if cp.get('query_fp16',False) else self.c['fp32_mac_cycles']
                if cp.get('readout_identity'):
                    matrices.append((projection_count,lang,classes,query_precision))
                elif cp['readout_extra_flops_per_row']:
                    query_width=width+int(cp.get('query_bias',False))
                    matrices.extend(((projection_count,k,width,self.c['fp32_mac_cycles']),
                                     (projection_count,query_width,classes,query_precision),
                                     (output_count,width,lang,self.c['fp32_mac_cycles'])))
                    fusion=(query_width,lang,classes,self.c['fp32_mac_cycles'])
                else:
                    matrices.extend(((projection_count,k,classes,query_precision),
                                     (output_count,k,lang,self.c['fp32_mac_cycles'])))
                    fusion=(k,lang,classes,self.c['fp32_mac_cycles'])
                if not cp.get('readout_identity') and depth not in self.prepared_readouts:
                    # Text-dependent composition is not a free preprocessing
                    # constant: once per invocation, before its first use.
                    self.prepared_readouts.add(depth)
                    matrices.insert(0,fusion)
                    self.stats['query_projection_preparations']+=1
                for matrix_rows,inner,columns,precision in matrices:
                    if not matrix_rows:continue
                    flops+=2*matrix_rows*inner*columns
                    tiles=math.ceil(matrix_rows/self.c['array_rows'])*math.ceil(columns/self.c['array_cols'])
                    local_arrays=np.zeros(self.c['arrays'],dtype=np.int64)
                    duration=inner*precision+self.c['array_rows']+self.c['array_cols']-2
                    initiation=inner*precision if self.c['array_wave_pipeline'] else 0
                    total+=int(self.memory.core.matrix_tiles(local_arrays,tiles,duration,0,initiation))
                    bytes_+=(inner*columns+matrix_rows*(inner+columns))*4
                    if precision==1 and cp.get('query_fp16',False):
                        cast_elements=inner*columns+matrix_rows*(inner+columns)
                        # Preserve the FP32 source traffic above, and charge
                        # explicit half staging plus result conversion. No
                        # free quantized weights or extra matrix resources.
                        bytes_+=cast_elements*4
                        cast_ops=cast_elements
                        vector_ops+=cast_ops;total+=math.ceil(cast_ops/self.c['vector_lanes'])
                        self.stats['diagnostic_query_fp16_flops']+=2*matrix_rows*inner*columns
                        self.stats['diagnostic_cast_elements']+=cast_elements
                ops=projection_count*(3*k+3*classes)+control_count*(3*classes+2*classes+self.c['predictor_window']+2)+output_count*3*lang
            elif not cp.get('readout_identity'):
                readout_flops=2*projection_count*k*width+projection_count*cp['readout_extra_flops_per_row']
                flops+=readout_flops
                tiles=math.ceil(projection_count/self.c['array_rows'])*math.ceil(width/self.c['array_cols'])
                total+=math.ceil(tiles/self.c['arrays'])*(k*self.c['fp32_mac_cycles']+self.c['array_rows']+self.c['array_cols']-2)
                bytes_+=k*width*4+projection_count*(k+width)*4
                if cp['readout_extra_flops_per_row']:
                    tiles=math.ceil(projection_count/self.c['array_rows'])*math.ceil(lang/self.c['array_cols'])
                    total+=math.ceil(tiles/self.c['arrays'])*(width*self.c['fp32_mac_cycles']+self.c['array_rows']+self.c['array_cols']-2)
                    bytes_+=width*lang*4+projection_count*(width+lang)*4
            if not query_mode:
                flops+=2*projection_count*lang*classes
                tiles=math.ceil(projection_count/self.c['array_rows'])*math.ceil(classes/self.c['array_cols'])
                total+=math.ceil(tiles/self.c['arrays'])*(lang*self.c['fp32_mac_cycles']+self.c['array_rows']+self.c['array_cols']-2)
            # Exact alias trajectories share normalization/TV work; distinct
            # trajectories still update independently even at one current row.
                ops=projection_count*(3*k+3*lang)+control_count*(3*lang+2*classes+self.c['predictor_window']+2)
            if self.c.get('predictor_conditioned_response',False):
                # Binary/threshold response bucket, flattened address, valid
                # support and prior fallback, then conservative response max.
                ops+=control_count*(self.c['predictor_response_bins']-1+6)
                total+=math.ceil(control_count/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
                self.stats['conditional_table_lookups']+=control_count
                self.stats['control_sram_read_bytes']+=control_count*4
            vector_ops+=ops;total+=math.ceil(ops/self.c['vector_lanes'])
            state_width=classes if query_mode else lang
            bytes_+=(projection_count*state_width+control_count*(3*state_width+2*classes))*4
            total+=math.ceil((count+control_count)/self.c['cam_lookups_per_cycle'])+self.c['cam_latency_cycles']
        total+=math.ceil(bytes_*self.c['clock_hz']/self.c['dram_bandwidth_bytes_per_second'])
        self.stats['diagnostic_cycles']+=total;self.stats['diagnostic_flops']+=flops
        self.stats['diagnostic_bytes']+=bytes_;self.stats['diagnostic_vector_ops']+=vector_ops
        self.cycle+=total
        if emitted:
            # Early semantic results have the same dense per-Gaussian output
            # contract as the full model. Aliased readouts may share compute,
            # but their individual output writes cannot disappear.
            output=self.g.events[-1]['outputs'][0]
            width=output['shape'][-1]*output['element_bytes']
            batch=max(1,self.c['output_sram_bytes']//self.c['output_buffers']//width)
            selected=np.asarray(emitted,dtype=np.int64)
            for begin in range(0,len(selected),batch):
                ids=selected[begin:begin+batch]
                self.memory.clock=max(self.memory.clock,self.cycle)
                self.cycle=max(self.cycle,self.store(output,ids,self.cycle))
            self.stats['early_output_rows']+=len(selected)
            self.stats['early_output_bytes']+=len(selected)*width
        return total

    def finish(self,coverage_meta=None):
        from .normalization_timing import release as release_normalization
        release_normalization(self)
        if self.c.get('finite_diagnostic_pipeline',False):
            from .diagnostic_pipeline import release
            release(self)
        if coverage_meta is not None:
            if self.coverage_preparation_meta is None:
                self.install_coverage_preparation(coverage_meta)
            elif coverage_meta!=self.coverage_preparation_meta:
                raise ValueError('Coverage preparation metadata changed')
            self.finish_coverage_preparation()
            self.cycle=max(self.cycle,self.coverage_preparation_ready)
            self.memory.clock=max(self.memory.clock,self.cycle)
        if self.coverage_rows is not None:
            cost=math.ceil(self.coverage_rows.scalar_ops/self.c['vector_lanes'])
            cost+=math.ceil(self.coverage_rows.metadata_bytes*self.c['clock_hz']/self.c['dram_bandwidth_bytes_per_second'])
            self.stats['coverage_row_metadata_ops']=self.coverage_rows.scalar_ops
            self.stats['coverage_row_metadata_bytes']=self.coverage_rows.metadata_bytes
            if self.c.get('coverage_layout_domains',False):
                self.stats['coverage_layout_domains']=len(self.coverage_rows.layout_domains)
                self.stats['coverage_layout_domain_reuses']=self.coverage_rows.layout_domain_reuses
            if self.c.get('coverage_link_domains',False):
                self.stats['coverage_link_domains']=len(self.coverage_rows.native_link_domains)
                self.stats['coverage_link_domain_reuses']=self.coverage_rows.native_link_domain_reuses
            self.stats['coverage_row_preparation_cycles']=cost
            self.stats['coverage_preparation_cycles']+=cost;self.cycle+=cost
        # Final dirty data must leave the accelerator before E2E completion.
        if self.c.get('event_local_completion',False):
            # Local dispatch barriers never retire an unrelated outstanding
            # result. The invocation join drains every real completion domain.
            self.cycle=max(self.cycle,self.stats['last_tile_completion'],
                           int(max(self.matrix)),self.vector)
        self.memory.clock=max(self.memory.clock,self.cycle)
        self.cycle=max(self.cycle,self.memory.finish())
        if self.c.get('vector_ready_queue',False):
            self.stats['vector_ready_queue_bytes']=self.vector_queue_storage_bytes
            self.stats['vector_ready_queue_peak']=self.vector_queue.peak
            self.stats['vector_queue_backpressure_cycles']=self.vector_queue.backpressure
            self.stats['vector_filled_gap_cycles']=self.vector_queue.hidden
        if self.activity is not None and self.stats['diagnostic_cycles'] and not self.c.get('finite_diagnostic_pipeline',False):
            raise NotImplementedError('Execution interval observation requires finite diagnostic programs')
        activity=self.activity.summarize(self.cycle,np.asarray(self.memory.dma_activity,dtype=np.int64).reshape(-1,2)) if self.activity is not None else None
        if activity is not None and activity['array_issue_lane_cycles']!=self.stats['array_occupied_cycles']:
            raise AssertionError('Matrix activity disagrees with issued work')
        if activity is not None:
            if activity['dram_service_overlap']['dram_service_ticks']!=self.memory.counts[2]*self.memory.burst_cycles:
                raise AssertionError('DDR activity disagrees with transferred bursts')
            activity['aggregate_coverage_preparation_outside_intervals']=self.stats.get('coverage_preparation_cycles',0)
        return dict(cycles=int(self.cycle),ms=self.cycle/self.c['clock_hz']*1000,
                    **({'copy_admission_observation':self.memory.observed_copy_admission()}
                       if self.c.get('observe_copy_admission',False) else {}),
                    event_profile=self.event_profile if self.c.get('profile_events',False) else None,
                    execution_intervals=activity,
                    memory_paths=self.memory.observed_paths(),
                    control_lowering_status=('not_used' if not self.stats['diagnostic_cycles'] else
                        ('finite_trace_microcode' if getattr(self,'feedback_published',False) else 'finite_observations_without_pilot_feedback')
                        if self.c.get('finite_diagnostic_pipeline',False) else 'aggregate_diagnostic_estimate'),
                    operand_timing_contract=('finite_tiles' if self.c.get('finite_operand_feeder',False) else 'legacy_unbounded_staging'),
                    statistics={k:int(v) for k,v in self.stats.items()},
                    memory_statistics={k:int(v) for k,v in self.memory.stats.items()},
                    resource_activity=dict(
                        array_issue_fraction=self.stats['array_occupied_cycles']/max(1,self.cycle*self.c['arrays']),
                        vector_issue_fraction=self.stats['vector_cycles']/max(1,self.cycle),
                        dram_service_fraction=(self.memory.stats['dram_read_bytes']+self.memory.stats['dram_write_bytes'])*self.c['clock_hz']/self.c['dram_bandwidth_bytes_per_second']/max(1,self.cycle),
                        diagnostic_activity_included=not self.stats['diagnostic_cycles'] or self.c.get('finite_diagnostic_pipeline',False)),
                    stage_cycles={k:int(v) for k,v in self.stages.items()})
