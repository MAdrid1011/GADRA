"""One finite A checkpoint descriptor on the existing shared datapaths."""
import math
import numpy as np

from .protocol import DIAGNOSTIC_FORK_DESCRIPTOR_BYTES,DIAGNOSTIC_ROW_LOOP_BYTES,SLOT_BYTES
from .graph import key,rows


class CheckpointFork:
    """Join every fixed-row region before the next dynamic row selection.

    Boundaries come from actual calls to progressive.checkpoint_program, not
    inferred future masks. The host holds the trace; the hardware owns only
    one reused result/dispatch/join descriptor, in the original control SRAM.
    """
    @staticmethod
    def terminal_regions(graph,existing):
        """Find fixed arithmetic tails ending in an actual indexed row store.

        This only proposes regions. The ordinary physical-alias, immutable
        selector, dependency, capacity and demand validator must admit each
        region before any row window is executed.
        """
        allowed=('aten.gelu.','aten.relu.','aten.add.','aten.sub.','aten.mul.',
                 'aten.div.','aten.to.','aten.clamp_min.','aten.linalg_vector_norm.')
        regions=[]
        for end,event in enumerate(graph.events):
            if event['kind']!='indexed_store' or len(event['inputs'])!=3:continue
            source=event['inputs'][2]
            if len(source['shape'])!=2:continue
            n=rows(source);begin=end
            while begin:
                previous=graph.events[begin-1]
                if (len(previous['outputs'])!=1 or len(previous['outputs'][0]['shape'])!=2
                    or rows(previous['outputs'][0])!=n or previous['stage']!=event['stage']):break
                if previous['kind']=='linear':begin-=1;break
                if previous['kind']!='aten' or not previous.get('op','').startswith(allowed):break
                begin-=1
            if begin==end or any(not (end<r['begin'] or begin>r['end']) for r in existing+regions):continue
            regions.append(dict(begin=begin,end=end))
        return regions

    def __init__(self,graph,config):
        self.starts={};self.ends={};self.floor=None;self.current=None
        regions=list(graph.maps.get('channel_checkpoints',[])) if config.get('channel_checkpoint_overlap',False) else []
        terminals=(self.terminal_regions(graph,regions) if config.get('stream_terminal_row_stores',False) else [])
        self.terminal_starts={r['begin'] for r in terminals}
        regions=sorted(regions+terminals,key=lambda r:r['begin'])
        if regions:
            previous=-1
            for region in regions:
                begin,end=region['begin'],region['end']
                if not (previous<begin<=end<len(graph.events)):
                    raise ValueError('Channel checkpoints must be disjoint ordered regions')
                for event in graph.events[begin:end+1]:
                    if event.get('op','').startswith(('aten.nonzero','aten.masked_select')) or (
                        (event.get('row_selector') or event.get('op')=='aten.index.Tensor')
                        and any(s.get('dtype')=='torch.bool' for s in event['inputs'][1:])):
                        raise ValueError('Dynamic row selection must follow the checkpoint join')
                self.starts[begin]=end;self.ends[end]=begin;previous=end
        self.row_windows=config.get('channel_checkpoint_row_windows',False)
        self.storage_bytes=(DIAGNOSTIC_FORK_DESCRIPTOR_BYTES+
            (DIAGNOSTIC_ROW_LOOP_BYTES if self.row_windows or terminals else 0)) if self.starts else 0

    def window_rows(self,sim,begin,live):
        """Admit only a fixed, independent row domain; never assume cache hits.

        Check every physical read/write overlap, including aliases of other
        versions. A write may intersect accesses of its own local row only.
        Unsupported or cross-row programs retain the original full-row order.
        This is static microprogram validation, not a runtime prediction.
        """
        end=self.starts[begin];domain=live[begin];n=len(domain)
        if (not self.row_windows and begin not in self.terminal_starts) or not n:return None
        barriers=set(sim.g.maps.get('diagnostic_control_barriers',{}))
        barriers.update(cp['after_event'] for cp in sim.g.maps.get('checkpoints',[]))
        if any(begin<=at<end for at in barriers):return None
        if any(not np.array_equal(live[j],domain) or j in getattr(live,'carry',{})
               or j in sim.column_epilogues for j in range(begin,end+1)):return None
        allowed=('aten.gelu.','aten.relu.','aten.add.','aten.sub.','aten.mul.',
                 'aten.div.','aten.to.','aten.clamp_min.','aten.linalg_vector_norm.')
        accesses=[];writes=[];footprints={}
        written_bases={sim.allocation_base(key(s)) for e in sim.g.events[begin:end+1]
                       for s in e['outputs']}
        for j in range(begin,end+1):
            event=sim.g.events[j];kind=event['kind']
            gather=kind=='aten' and event.get('op')=='aten.index.Tensor'
            if not (kind in ('linear','progressive_difference','progressive_update','progressive_observe','indexed_store')
                    or gather
                    or (kind=='aten' and event.get('op','').startswith(allowed))):return None
            if gather or kind=='indexed_store':
                inputs=event['inputs'];out=event['outputs'][0]
                mapping=sim.g.maps['indices'].get(j,[])
                if (len(inputs)!=(2 if gather else 3) or len(event['outputs'])!=1
                    or len(mapping)!=1):return None
                selector=inputs[1];ids=np.asarray(mapping[0])
                source=inputs[0] if gather else out
                if (len(source['shape'])!=2 or len(out['shape'])!=2
                    or source['shape'][1]!=out['shape'][1]
                    or ids.ndim!=1 or ids.dtype.kind not in 'iu'
                    or selector.get('dtype') not in ('torch.int32','torch.int64')
                    or selector['shape'] not in ([len(ids)],[len(ids),1])
                    or len(ids)!=(rows(out) if gather else rows(inputs[2]))
                    or (not gather and (len(inputs[2]['shape'])!=2
                        or inputs[2]['shape'][1]!=out['shape'][1]))
                    or sim.allocation_base(key(selector)) in written_bases
                    or np.any(ids<(-rows(source) if gather else 0))
                    or np.any(ids>=rows(source))):return None
                # The selector is a completed, immutable input of this join.
                # Its actual read/decode and all gather/store traffic remain.
            refs=sim.g.inputs_for(j,domain)
            outputs=[]
            for k,spec in enumerate(event['outputs']):
                ids=domain
                if (kind in ('progressive_update','indexed_store') and k==0) or (kind=='progressive_observe' and k>=2):
                    ids=np.asarray(sim.g.maps['indices'][j][0])[domain]
                outputs.append((spec,ids))
            for position,(spec,ids) in enumerate(refs+outputs):
                if not 1<=len(spec['shape'])<=2 or any(s<0 for s in spec['stride']):return None
                width=((spec['shape'][-1]-1)*spec['stride'][-1]+1)*spec['element_bytes']
                starts=sim.g.address_starts(spec,ids)
                # A whole RHS/LUT can coincidentally have N rows. Those N
                # reads belong to every consumer, not one read per local row.
                local=(position>=len(refs) or not (
                    spec['role']=='weight' or
                    (kind=='linear' and position in (1,2)) or
                    (kind=='progressive_observe' and position in (5,6,7)) or
                    (kind=='aten' and not gather and rows(spec)!=rows(event['outputs'][0]))))
                access=(sim.allocation_base(key(spec)),starts,width,local)
                accesses.append(access)
                if any(spec is out for out,_ in outputs):writes.append(access)
                if spec['role']!='feature':continue
                identity=(key(spec),spec['address'],tuple(spec['shape']),tuple(spec['stride']))
                footprints[identity]=(width if len(ids)==n else 0,spec['bytes'] if len(ids)!=n else 0)
        # No two local rows may write overlapping bytes (including repeated
        # global destination IDs). Sector-sharing stores still use write masks.
        for base,starts,width,_ in writes:
            if len(starts)!=n:return None
            order=np.argsort(starts);sorted_starts=starts[order]
            if np.any(sorted_starts[1:]<sorted_starts[:-1]+width):return None
            for other_base,other,other_width,local in accesses:
                if base!=other_base or not len(other):continue
                if other.max()+other_width<=sorted_starts[0] or other.min()>=sorted_starts[-1]+width:continue
                if not local or len(other)!=n:return None
                # First/last write row intersecting each read/write interval.
                first=np.searchsorted(sorted_starts,other-width,side='right')
                last=np.searchsorted(sorted_starts,other+other_width,side='left')
                touched=last>first
                if np.any(last[touched]-first[touched]!=1):return None
                if np.any(order[first[touched]]!=np.flatnonzero(touched)):return None
        width=sum(p[0] for p in footprints.values());fixed=sum(p[1] for p in footprints.values())
        capacity=min(sim.c['feature_sram_bytes'],sim.c['cam_entries']*SLOT_BYTES)-fixed
        if not width or capacity<width*sim.c['array_rows']:return None
        group=min(capacity//width,sim.c['queue_entries']*sim.c['array_rows'])
        return max(sim.c['array_rows'],group//sim.c['array_rows']*sim.c['array_rows'])

    def schedule(self,sim,begin,live,group):
        """Reuse one finite loop descriptor and all ordinary data stores."""
        end=self.starts[begin];domain=live[begin]
        sim.stats['channel_checkpoint_windowed_regions']+=1
        for offset in range(0,len(domain),group):
            # Actual row-ID arrays are still read by the native indexed ops.
            # Only cursor/end/PC live in the additional 12B control fields.
            cost=2*(math.ceil(DIAGNOSTIC_ROW_LOOP_BYTES/(4*sim.c['cam_lookups_per_cycle']))+sim.c['cam_latency_cycles'])
            sim.memory.clock=max(sim.memory.clock,sim.cycle)+cost
            sim.cycle=max(sim.cycle,sim.memory.clock)
            sim.stats['channel_checkpoint_control_cycles']+=cost
            sim.stats['control_sram_write_bytes']+=DIAGNOSTIC_ROW_LOOP_BYTES
            sim.stats['control_sram_read_bytes']+=DIAGNOSTIC_ROW_LOOP_BYTES
            sim.stats['channel_checkpoint_row_windows']+=1
            selected=domain[offset:offset+group];last=offset+group>=len(domain)
            for j in range(begin,end+1):yield j,sim.g.events[j],selected,last

    def charge(self,sim,write):
        cost=math.ceil(self.storage_bytes/(4*sim.c['cam_lookups_per_cycle']))+sim.c['cam_latency_cycles']
        sim.memory.clock=max(sim.memory.clock,sim.cycle)+cost
        sim.cycle=max(sim.cycle,sim.memory.clock)
        sim.stats['channel_checkpoint_control_cycles']+=cost
        sim.stats['control_sram_write_bytes' if write else 'control_sram_read_bytes']+=self.storage_bytes

    def enter(self,sim,index):
        if index in self.starts and self.current is None:
            # Previous compaction, output ownership and all issued work drain
            # before a new active-row domain can be used by this descriptor.
            sim.cycle=max(sim.cycle,sim.memory.clock,sim.stats['last_tile_completion'],int(max(sim.matrix)),sim.vector)
            self.charge(sim,True)
            self.current=index;self.floor=sim.cycle
            sim.stats['channel_checkpoint_programs']+=1
            sim.stats['channel_checkpoint_descriptor_peak_bytes']=self.storage_bytes
        if self.current is not None and not self.current<=index<=self.starts[self.current]:
            raise ValueError('Checkpoint descriptor reused before its join')
        return self.current is not None

    def join(self,sim,index,last_chunk):
        if self.current is None or index!=self.starts[self.current] or not last_chunk:return
        # Includes every wide-output store and narrow/control branch, not
        # only the last lexical operation. No captured future row count may
        # escape this boundary before those real producers complete.
        sim.cycle=max(sim.cycle,sim.memory.clock,sim.stats['last_tile_completion'],int(max(sim.matrix)),sim.vector)
        self.charge(sim,False)
        self.current=None;self.floor=None

    def boundaries(self):
        return set(self.ends)|{begin-1 for begin in self.starts if begin}
