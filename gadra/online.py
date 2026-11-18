"""Checkpoint-committed decisions and causal task-demand updates.

Captured values are supplied one checkpoint at a time. Only completed pilot
responses may describe later processing. No final stop-depth vector is an
input to this execution path.
"""
from dataclasses import dataclass
import math
import numpy as np

from .graph import DemandPlan
from .protocol import DMA_BURST_BYTES


@dataclass(frozen=True)
class Proposal:
    depth: int
    positions: np.ndarray
    normalized: np.ndarray
    variation: np.ndarray
    gap: np.ndarray
    exits: np.ndarray
    stop_depths: np.ndarray
    unsupported_matches: np.ndarray


class DecisionState:
    def __init__(self, predictor, contexts, witnesses):
        self.predictor=predictor
        self.contexts=np.asarray(contexts)
        self.witnesses=np.asarray(witnesses,dtype=bool)
        self.depth_count=len(predictor.table)
        self.stops=np.full(len(contexts),self.depth_count-1,np.int16)
        self.pending=~self.witnesses.copy()
        self.previous=None
        self.previous_depth=np.full(len(contexts),-1,np.int16)
        self.proposal=None
        self.events=[]
        self.last_commit_cycle=-1
        self.next_depth=0
        self.unsupported_positions=np.zeros(len(contexts),bool)

    def propose(self, depth, current_scores, completed_records):
        if self.proposal is not None:
            raise ValueError('Commit the preceding checkpoint decision first')
        if not 0<=depth<self.depth_count-1:
            raise ValueError('The final native output is not an exit checkpoint')
        if depth!=self.next_depth:
            raise ValueError('Observe checkpoints in committed native program order')
        positions=np.flatnonzero(self.pending)
        score=np.asarray(current_scores,dtype=np.float32)
        if (score.ndim!=2 or len(score)!=len(positions) or score.shape[1]<2
                or not np.isfinite(score).all()):
            raise ValueError('A checkpoint needs finite current scores for pending positions')
        scale=np.maximum(np.abs(score).sum(1,keepdims=True),np.finfo(np.float32).tiny)
        normalized=score/scale
        if self.previous is None:
            self.previous=np.zeros((len(self.contexts),score.shape[1]),np.float32)
        if score.shape[1]!=self.previous.shape[1]:
            raise ValueError('The query set must remain fixed within one inference')
        if depth and np.any(self.previous_depth[positions]!=depth-1):
            raise ValueError('A pending position is missing its preceding committed observation')
        variation=(np.abs(normalized-self.previous[positions]).sum(1)
                   if depth else np.zeros(len(positions),np.float32))
        top=np.partition(normalized,-2,axis=1)[:,-2:]
        gap=top.max(1)-top.min(1)
        bins=self.predictor.config['predictor_bins']
        tv_edges,gap_edges=self.predictor.edges[depth]
        cell=np.searchsorted(tv_edges,variation)*bins+np.searchsorted(gap_edges,gap)
        local=self.predictor.table[depth][cell]
        response=np.full(len(positions),-np.inf,np.float32)
        if completed_records is not None:
            keys,values,groups=completed_records
            winner=score.argmax(1)
            at=np.asarray(groups)[positions]
            matching=keys[depth,at]==winner[:,None]
            response=np.where(matching,values[depth,at],-np.inf).max(1)
        matched=np.isfinite(response)
        estimate=local.copy()
        unsupported=np.zeros(len(positions),bool)
        if self.predictor.response_table:
            response_cell=(cell*self.predictor.config['predictor_response_bins']+
                np.searchsorted(self.predictor.response_edges[depth],response))
            conditional=self.predictor.response_table[depth][response_cell]
            # An unsupported selected conditional cell remains +inf.
            estimate[matched]=conditional[matched]
            tolerance=self.predictor.config['semantic_tolerance']
            unsupported=(matched&(~np.isfinite(conditional))&
                         (np.maximum(local,response)<=tolerance))
        estimate=np.maximum(estimate,response)
        eligible=(estimate<=self.predictor.config['semantic_tolerance'])
        if depth<self.predictor.config['predictor_window']-1:
            eligible[:]=False
            unsupported[:]=False
        exits=positions[eligible]
        proposed=self.stops.copy()
        proposed[exits]=depth
        self.proposal=Proposal(depth,positions,normalized,variation,gap,exits,proposed,
                               positions[unsupported])
        return self.proposal

    def commit(self, cycle):
        p=self.proposal
        if p is None:
            raise ValueError('There is no completed checkpoint decision to commit')
        if int(cycle)!=cycle or cycle<self.last_commit_cycle or cycle<0:
            raise ValueError('Checkpoint commits must have ordered nonnegative integer cycles')
        self.previous[p.positions]=p.normalized
        self.previous_depth[p.positions]=p.depth
        self.stops[p.exits]=p.depth
        self.pending[p.exits]=False
        self.unsupported_positions[p.unsupported_matches]=True
        self.events.append(dict(kind='checkpoint_commit',depth=p.depth,
            cycle=int(cycle),positions=p.exits.tolist(),observed_positions=len(p.positions)))
        self.last_commit_cycle=int(cycle)
        self.next_depth=p.depth+1
        self.proposal=None
        return p.exits


def request_seeds(graph, pending, next_depth):
    """Known pending outputs and their future diagnostic inputs, without future decisions."""
    positions=np.flatnonzero(pending)
    if not len(positions):
        return []
    checkpoints=graph.maps['checkpoints']
    final=checkpoints[-1]
    seeds=[(final['spec'],np.unique(np.asarray(final['mapping'])[positions]))]
    for cp in checkpoints[next_depth:-1]:
        seeds.extend((spec,np.unique(np.asarray(mapping)[positions]))
                     for spec,mapping in zip(cp['readout_inputs'],cp['readout_maps']))
    return seeds


def executed_segment(plan, start, end):
    """Only this admitted native-program interval enters the execution ledger."""
    live=[ids.copy() if start<=i<end else np.empty(0,np.int64)
          for i,ids in enumerate(plan)]
    carry={i:(old,new,tokens.copy()) for i,(old,new,tokens) in plan.carry.items()
           if start<=i<end}
    return DemandPlan(live,carry)


def union_completed(first, second):
    carry=dict(first.carry)
    for i,(old,new,tokens) in second.carry.items():
        if i in carry:
            a,b,previous=carry[i]
            if (a,b)!=(old,new):
                raise ValueError('A completed storage inheritance changed identity')
            tokens=np.union1d(previous,tokens)
        carry[i]=(old,new,tokens)
    return DemandPlan([np.union1d(a,b) for a,b in zip(first,second)],carry)


def cancellation_work(graph, before, after, dropped_positions, next_depth, observed_positions):
    """Reference decrements for removed demands, including already-issued producers."""
    removed=before.minus(after)
    added=after.minus(before)
    if any(len(ids) for ids in added) or added.carry:
        raise ValueError('Cancellation cannot introduce new producer requests')
    observed=graph.maps['checkpoints'][next_depth-1]
    roots=len(observed_positions)*len(observed.get('readout_inputs',[]))
    for cp in graph.maps['checkpoints'][next_depth:]:
        roots+=len(dropped_positions)*max(1,len(cp.get('readout_inputs',[])))
    edges=0
    for i,ids in enumerate(removed):
        if len(ids):
            edges+=sum(len(source_rows) for spec,source_rows in graph.inputs_for(i,ids))
    edges+=sum(len(tokens) for old,new,tokens in removed.carry.values())
    # A root/map read and reference/bitmap update use explicit 32-bit records.
    updates=roots+edges
    return removed,dict(root_decrements=roots,dependency_decrements=edges,
        read_bytes=updates*8,write_bytes=updates*4,integer_operations=updates*3)


def apply_cancellation_cost(sim, work):
    """Charge metadata transfers and reference updates before the next issue interval."""
    m=sim.memory
    ready=max(sim.cycle,m.clock)
    for write,nbytes in ((False,work['read_bytes']),(True,work['write_bytes'])):
        for _ in range(math.ceil(nbytes/DMA_BURST_BYTES)):
            ready=max(ready,int(m.core.transaction(ready,write,m.state,m.pending,m.counts,
                m.burst_cycles,sim.c['dram_latency_cycles'],DMA_BURST_BYTES,m.dma_activity)))
    operations=work['integer_operations']
    ready=sim.issue_vector(ready,operations,operator='dependency_reference_update')
    sim.stats['vector_cycles']+=operations
    sim.stats['dependency_reference_updates']+=work['root_decrements']+work['dependency_decrements']
    sim.stats['dependency_state_dram_read_bytes']+=math.ceil(work['read_bytes']/DMA_BURST_BYTES)*DMA_BURST_BYTES
    sim.stats['dependency_state_dram_write_bytes']+=math.ceil(work['write_bytes']/DMA_BURST_BYTES)*DMA_BURST_BYTES
    sim.cycle=max(sim.cycle,ready)
    m.clock=max(m.clock,sim.cycle)


def execute_online(sim, predictor, observations, witnesses, pilot_plan):
    """Execute complete pilots, then admit work using committed checkpoint decisions."""
    graph=sim.g
    context=graph.maps['response_context']
    state=DecisionState(predictor,context['mapping'],witnesses)
    # Fixed pilot membership is derived from input context IDs, not output quality.
    sim.execute(pilot_plan,control=(observations,state.stops,witnesses,'pilot'))
    sim.audit_events.append(dict(kind='native_outputs_commit',cycle=int(sim.cycle),
        positions=np.flatnonzero(witnesses).tolist(),depth=len(graph.maps['checkpoints'])-1))
    from .diagnostic_pipeline import complete_feedback
    complete_feedback(sim,observations,witnesses)
    if not getattr(sim,'feedback_published',False):
        raise ValueError('Online execution requires committed pilot responses')
    completed=pilot_plan
    plan=graph.closure(request_seeds(graph,state.pending,0))
    start=0
    for depth,cp in enumerate(graph.maps['checkpoints'][:-1]):
        if not state.pending.any():
            break
        end=cp['after_event']+1
        if end<start:
            raise ValueError('Checkpoint boundaries are not in native program order')
        remaining=plan.minus(completed)
        if any(len(ids) for ids in remaining[:start]) or any(i<start for i in remaining.carry):
            raise ValueError('A pending request needs a producer outside the completed native prefix')
        sim.execute(remaining,control=(observations,state.stops,witnesses,'remaining'),
                    start=start,end=end,observe_checkpoints=False)
        completed=union_completed(completed,executed_segment(remaining,start,end))
        # The proposal can use this checkpoint and committed history only.
        ids=np.flatnonzero(state.pending)
        proposal=state.propose(depth,observations.get('control_scores',observations['scores'])[depth,ids],
                               sim.completed_response_records)
        sim.online_lookup=(depth,proposal.positions,proposal.variation,proposal.gap)
        sim.diagnostics(observations,proposal.stop_depths,witnesses,
                        depths=[depth],phase='remaining')
        sim.online_lookup=None
        commit_cycle=max(sim.cycle,sim.memory.clock)
        dropped=state.commit(commit_cycle)
        sim.audit_events.extend(state.events[-1:])
        after=graph.closure(request_seeds(graph,state.pending,depth+1))
        removed,work=cancellation_work(graph,plan,after,dropped,depth+1,proposal.positions)
        apply_cancellation_cost(sim,work)
        # Admission history is immutable. Cancellation never subtracts executed work.
        for i,ids in enumerate(removed):
            cancellable=np.setdiff1d(ids,completed[i],assume_unique=True)
            if len(cancellable):
                sim.audit_events.append(dict(kind='task_cancel',event=i,
                    cycle=int(sim.cycle),positions=cancellable.tolist(),
                    decision_cycle=int(commit_cycle),depth=depth))
        plan=after
        start=end
    remaining=plan.minus(completed)
    if any(len(ids) for ids in remaining[:start]) or any(i<start for i in remaining.carry):
        raise ValueError('Final outputs require a producer outside the completed native prefix')
    sim.execute(remaining,control=(observations,state.stops,witnesses,'remaining'),
                start=start,end=len(graph.events),observe_checkpoints=False)
    completed=union_completed(completed,executed_segment(remaining,start,len(graph.events)))
    sim.audit_events.append(dict(kind='native_outputs_commit',cycle=int(sim.cycle),
        positions=np.flatnonzero(state.pending).tolist(),depth=len(graph.maps['checkpoints'])-1))
    state.pending[:]=False
    main_flops=sum(graph.flops(event['event'],np.asarray(event['positions'],np.int64))
        for event in sim.audit_events if event['kind']=='native_issue')
    return state,main_flops
