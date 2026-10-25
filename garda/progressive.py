"""Experimental causal channel checkpoints using completed context consumers.

The shared-parent consumer runs its actual complete MLP first. Another row
starts from that result and replaces hidden-channel contributions incrementally.
Only independent-scene calibration and completed consumer data enter a stop.
This is a candidate checkpoint/readout refinement, not a demonstrated hardware
speedup or a replacement for the design's supply dependencies.
"""
from contextlib import contextmanager
import fcntl
import json
import pickle
from pathlib import Path
import types
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .factorized import AncestorAffine,context_banks,paired
from .predictor import DemandLUT,demand_features,remaining_correction
from .workloads import WORKLOADS,load_case


@contextmanager
def checkpoint_program():
    """Fixed-row checkpoint region; compaction occurs only after its join.

    The software reference preserves its ordinary stream order. Timing capture
    records the boundary so the original shared units can issue ready branches
    independently, with a single finite controller descriptor.
    """
    yield


def mapped_linear(state,weight,bias,ids):
    """Original affine arithmetic with a direct indexed-supply boundary.

    CUDA reference deliberately keeps the original gather/GEMM call. Hardware
    capture reuses indexed_source: read these IDs, load the actual native
    rows through the finite feeder, and write only the affine result. No
    materialized selected-row matrix is required by the mathematical DAG.
    """
    return F.linear(state[ids],weight,bias)


def observe_reference(raw,ids,mapping,previous,delta,tv_edge,margin_edge,table,response,response_winner,
                      window,bins,depth,depths,tolerance,force_full=False):
    """Readable arithmetic/control reference also used by functional replay."""
    score=F.normalize(raw,dim=1)
    if depth:delta[ids,(depth-1)%(window-1)]=(score-previous[ids]).abs().sum(1)
    previous[ids]=score
    keep=torch.ones(len(ids),device=ids.device,dtype=torch.bool)
    if not force_full and window-1<=depth<depths-1:
        tv=delta[ids].mean(1);top=score.topk(2,dim=1).values;margin=top[:,0]-top[:,1]
        cell=torch.bucketize(tv.contiguous(),tv_edge)*bins+torch.bucketize(margin.contiguous(),margin_edge)
        parent=mapping[ids]
        applicable=response_winner[parent]==score.argmax(1)
        feedback=torch.where(applicable,response[parent],torch.inf)
        keep=torch.maximum(table[cell],feedback)>tolerance
    return score,keep


def channel_checkpoints(width,chunk,growth=1):
    """Static channel boundaries; growth=1 preserves uniform checkpoints.

    A geometric schedule observes small prefixes early, then amortizes the
    same state/control pipeline over larger refinements. Every channel is
    still executed exactly once for a full-depth consumer.
    """
    for name,value in (('width',width),('chunk',chunk),('growth',growth)):
        if not isinstance(value,int) or isinstance(value,bool) or value<1:
            raise ValueError(f'Channel {name} must be a positive integer')
    boundaries=[];end=chunk
    while end<width:
        boundaries.append(end)
        end=end+chunk if growth==1 else end*growth
    return np.asarray([*boundaries,width],np.int64)


def context_level(value):
    if not isinstance(value,int) or isinstance(value,bool) or not 1<=value<=4:
        raise ValueError('Channel context level must be an integer from 1 to 4')
    return value


def consumer_order(keys):
    """Stable finite-key order; hardware capture reserves real sort scratch."""
    return keys.argsort(stable=True)


def select_consumers(state,mapping,groups,center=False,fused_distance=False):
    """Choose real consumers from ready inputs, without semantic labels.

    The optional center is the midpoint of each input-feature bounding box.
    Two nearest actual rows supply both the seed and completed feedback.
    Sorting, CSR min/max, distances and integer selection execute normally.
    """
    ids=torch.arange(len(mapping),device=mapping.device)
    order=ids
    if center:
        grouped=consumer_order(mapping)
        offsets=F.pad(torch.bincount(mapping,minlength=groups).cumsum(0),(1,0))
        values=state[grouped].float()
        low=torch.segment_reduce(values,'min',offsets=offsets,axis=0)
        high=torch.segment_reduce(values,'max',offsets=offsets,axis=0)
        midpoint=(low+high)*.5
        if fused_distance:
            from .progressive_triton import center_distance
            distance=center_distance(state,midpoint,mapping)
        else:distance=(state.float()-midpoint[mapping]).norm(dim=1)
        order=consumer_order(distance)
    parents=mapping[order] if center else mapping
    first=torch.full((groups,),len(mapping),device=mapping.device,dtype=torch.int64)
    first.scatter_reduce_(0,parents,ids,reduce='amin')
    second=torch.full_like(first,len(mapping))
    second.scatter_reduce_(0,parents,torch.where(ids!=first[parents],ids,len(mapping)),reduce='amin')
    second=second[second<len(mapping)]
    return (order[first],order[second]) if center else (first,second)


def calibration_consumers(observation,mapping,center=False):
    """Recover only actually selected calibration consumers and their owners."""
    _,first,inverse=np.unique(mapping,return_index=True,return_inverse=True)
    if center:
        if 'seed_rows' not in observation or 'responder_rows' not in observation:
            raise ValueError('Center calibration requires actual consumer identities')
        first=np.asarray(observation['seed_rows'],dtype=np.int64)
        responders=np.asarray(observation['responder_rows'],dtype=np.int64)
        if (first.ndim!=1 or responders.ndim!=1 or len(first)!=len(np.unique(inverse))
            or np.any(first<0) or np.any(first>=len(mapping))
            or np.any(responders<0) or np.any(responders>=len(mapping))
            or not np.array_equal(inverse[first],np.arange(len(first)))
            or not np.array_equal(inverse[responders],np.flatnonzero(np.bincount(inverse)>1))
            or np.any(np.isin(responders,first))):
            raise ValueError('Invalid completed-consumer calibration identities')
    else:
        ids=np.arange(len(mapping));second=np.full(len(first),len(mapping),np.int64)
        np.minimum.at(second,inverse,np.where(ids!=first[inverse],ids,len(mapping)))
        responders=second[second<len(mapping)]
    return first,responders,inverse


def channel_config(policy=None):
    config=json.loads(Path('configs/edge_reuse.json').read_text())
    config.update(witness_stride=1,predictor_shared_support=False,predictor_quantile=1.,
                  correction_horizon='suffix',class_repair_tolerance=0.)
    if policy is not None:
        supported={'predictor_quantile','correction_horizon','class_repair_tolerance',
                   'predictor_max_class_error','predictor_scene_risk_budget','predictor_scene_risk_candidates',
                   'predictor_window','channel_state_fp16',
                   'channel_query_transposed','channel_center_consumers','channel_center_fused_distance','channel_indexed_linear',
                   'channel_context_level','channel_checkpoint_growth'}
        if set(policy)-supported:raise ValueError(f'Unsupported channel policy keys: {sorted(set(policy)-supported)}')
        config.update(policy)
    if not isinstance(config['predictor_window'],int) or config['predictor_window']<2:
        raise ValueError('Channel trajectory window must contain at least two checkpoints')
    for flag in ('channel_state_fp16','channel_query_transposed','channel_center_consumers','channel_center_fused_distance','channel_indexed_linear'):
        if not isinstance(config.get(flag,False),bool):raise ValueError(f'{flag} must be boolean')
    if config.get('channel_center_fused_distance',False) and not config.get('channel_center_consumers',False):
        raise ValueError('Fused center distance requires center consumer selection')
    context_level(config.get('channel_context_level',2))
    channel_checkpoints(1,1,config.get('channel_checkpoint_growth',1))
    candidates=config.get('predictor_scene_risk_candidates','repair_quantile')
    if candidates not in ('repair_quantile','supported'):
        raise ValueError('Scene risk candidates must be repair_quantile or supported')
    if candidates!='repair_quantile' and 'predictor_scene_risk_budget' not in config:
        raise ValueError('Supported scene risk candidates require a scene risk budget')
    return config


def calibrated_lut(dataset,variant,root,policy=None,channel_chunk=512):
    config=channel_config(policy)
    datasets=[d for d in ('scannet','scannetpp','matterport3d') if d!=dataset]
    training=[dict(np.load(Path(root)/f'Chorus_{d}_{variant}_progressive.npz')) for d in datasets]
    for observation in training:
        if int(observation.get('context_level',2))!=config.get('channel_context_level',2):
            raise ValueError('Different completed-consumer context level')
        if bool(observation.get('state_fp16',False))!=config.get('channel_state_fp16',False):
            raise ValueError('Different channel state calibration precision')
        if bool(observation.get('query_transposed',False))!=config.get('channel_query_transposed',False):
            raise ValueError('Different channel query preparation layout')
        if bool(observation.get('center_consumers',False))!=config.get('channel_center_consumers',False):
            raise ValueError('Different channel consumer selection')
        if bool(observation.get('center_fused_distance',False))!=config.get('channel_center_fused_distance',False):
            raise ValueError('Different center distance arithmetic')
        if bool(observation.get('indexed_linear',False))!=config.get('channel_indexed_linear',False):
            raise ValueError('Different indexed affine arithmetic')
        channels=observation.get('checkpoint_channels')
        if channels is None:
            if channel_chunk!=512 or config.get('channel_checkpoint_growth',1)!=1:
                raise ValueError('Channel metadata missing; recapture calibration')
        elif not np.array_equal(channels,channel_checkpoints(int(channels[-1]),channel_chunk,config.get('channel_checkpoint_growth',1))):
            raise ValueError('Different channel calibration granularity')
    if 'predictor_scene_risk_budget' in config:config['predictor_edge_fp32']=True
    lut=DemandLUT(training,config)
    channels=[o.get('checkpoint_channels') for o in training]
    if any(c is not None for c in channels):
        if any(c is None or not np.array_equal(c,channels[0]) for c in channels):
            raise ValueError('Different independent channel checkpoint domains')
        lut.checkpoint_channels=channels[0].copy()
    if 'predictor_scene_risk_budget' in config:
        contexts=[]
        for d in datasets:
            geometry=pickle.loads(Path(f'results/raw/Chorus_{d}_s0_maps.pkl').read_bytes())
            suffix=f'enc{config.get("channel_context_level",2)}.block0'
            contexts.append(next(cp['mapping'] for cp in geometry['checkpoints'] if cp['name'].endswith(suffix)))
        compile_scene_risk(lut,training,contexts)
    return lut


def compile_scene_risk(lut,training,contexts):
    """Compile existing LUT actions under each independent scene's risk budget.

    Greedily admit the cell with greatest mean processed-channel saving whose
    *earliest-exit* predictions satisfy both scene budgets. Only calibration
    labels are read; runtime still performs the original table lookup and
    completed-second-consumer response check. No per-position exit map is kept.
    The supported mode lets all populated cells compete directly for the
    shared risk budget, without an additional repair-quantile prerequisite.
    """
    c=lut.config;budget=c['predictor_scene_risk_budget']
    if not 0<=budget<=1:raise ValueError('Scene risk budget must be in [0,1]')
    if c.get('predictor_max_class_error') is not None:
        raise ValueError('Choose cell risk or scene risk, not both')
    candidates=c.get('predictor_scene_risk_candidates','repair_quantile')
    if candidates not in ('repair_quantile','supported'):
        raise ValueError('Scene risk candidates must be repair_quantile or supported')
    if len(training)!=len(contexts) or not training:raise ValueError('Independent scene mappings required')
    tolerance=min(c['semantic_tolerance'],c['class_repair_tolerance'])
    depth_count=len(lut.table);bins=c['predictor_bins']
    widths=np.asarray(training[0]['checkpoint_channels'])
    if len(widths)!=depth_count or np.any(np.diff(widths)<=0):raise ValueError('Invalid channel checkpoints')
    width=widths[-1]
    scenes=[]
    for obs,mapping in zip(training,contexts):
        if 'output_winner' not in obs or 'output_regret' not in obs:
            raise ValueError('Scene risk requires actual emitted-prefix labels; recapture calibration')
        if not np.array_equal(obs['checkpoint_channels'],widths):raise ValueError('Different channel checkpoints')
        n=len(mapping)
        if obs['winner'].shape!=(depth_count,n):raise ValueError('Calibration and mapping domains differ')
        first,responders,inverse=calibration_consumers(obs,mapping,c.get('channel_center_consumers',False))
        pilot=np.zeros(n,bool);pilot[first]=True;pilot[responders]=True
        completed={k:(v[:,responders] if v.ndim>=2 and v.shape[1]==n else v) for k,v in obs.items()}
        response=np.full((depth_count,len(first)),np.inf,np.float32)
        keys=np.full(response.shape,-1,np.int64)
        response[:,inverse[responders]]=remaining_correction(completed,c['correction_horizon'])
        keys[:,inverse[responders]]=obs['winner'][:,responders]
        applicable=keys[:,inverse]==obs['winner']
        feedback=np.where(applicable,response[:,inverse],np.inf)
        features=demand_features(obs,c)
        cells=np.stack([np.searchsorted(a,features[0][d])*bins+np.searchsorted(b,features[1][d])
                        for d,(a,b) in enumerate(lut.edges)])
        scenes.append(dict(n=n,pilot=pilot,cells=cells,feedback=feedback,
            stop=np.full(n,depth_count-1,np.int64),
            error_count=int((obs['output_winner'][-1]!=obs['target']).sum()),
            regret_count=int((obs['output_regret'][-1]>c['semantic_tolerance']).sum()),
            error=obs['output_winner']!=obs['target'][None,:],
            regret=obs['output_regret']>c['semantic_tolerance']))
    actions=[]
    for depth in range(c['predictor_window']-1,depth_count-1):
        eligible=(np.isfinite(lut.table[depth]) if candidates=='supported'
                  else lut.table[depth]<=tolerance)
        for cell in np.flatnonzero(eligible):
            selections=[]
            for scene in scenes:
                mask=scene['cells'][depth]==cell
                if mask.sum()<c['predictor_min_samples']:break
                selections.append(np.flatnonzero(mask&~scene['pilot']&(scene['feedback'][depth]<=tolerance)))
            else:actions.append((depth,cell,selections))
    candidate_actions=len(actions)
    enabled=set()
    while actions:
        best=None
        for at,(depth,cell,selections) in enumerate(actions):
            gain=0.;valid=True;changes=[]
            for scene,selected in zip(scenes,selections):
                selected=selected[depth<scene['stop'][selected]]
                stop=scene['stop']
                error=scene['error_count']+scene['error'][depth,selected].sum()-scene['error'][stop[selected],selected].sum()
                regret=scene['regret_count']+scene['regret'][depth,selected].sum()-scene['regret'][stop[selected],selected].sum()
                if error>budget*scene['n'] or regret>budget*scene['n']:valid=False;break
                gain+=float((widths[stop[selected]]-widths[depth]).sum())/(scene['n']*width)
                changes.append(selected)
            if valid and gain>0 and (best is None or gain>best[0]):best=(gain,at,changes)
        if best is None:break
        _,at,changes=best;depth,cell,_=actions.pop(at);enabled.add((depth,cell))
        for scene,selected in zip(scenes,changes):
            stop=scene['stop']
            scene['error_count']+=int(scene['error'][depth,selected].sum()-scene['error'][stop[selected],selected].sum())
            scene['regret_count']+=int(scene['regret'][depth,selected].sum()-scene['regret'][stop[selected],selected].sum())
            stop[selected]=depth
    for depth,table in enumerate(lut.table):
        for cell in range(len(table)):
            if (depth,cell) not in enabled:table[cell]=np.inf
            elif candidates=='supported':
                # Compile the selected action into the existing scalar LUT;
                # completed-consumer repair still passes its original gate.
                table[cell]=min(table[cell],tolerance)
    reports=[]
    for scene in scenes:
        ids=np.arange(scene['n']);stop=scene['stop']
        reports.append(dict(rows=scene['n'],class_error=float(scene['error'][stop,ids].mean()),
            regret_error=float(scene['regret'][stop,ids].mean()),
            removed_head_channel_fraction=float(1-widths[stop].sum()/(scene['n']*width))))
    if any(r['class_error']>budget or r['regret_error']>budget for r in reports):
        raise ValueError('Scene risk compilation cannot meet the independent calibration budget')
    lut.risk_calibration=dict(scope='Independent scenes; greedy earliest-exit LUT cell selection',
                              enabled_actions=len(enabled),scenes=reports)
    if candidates!='repair_quantile':
        lut.risk_calibration.update(candidate_policy=candidates,candidate_actions=candidate_actions)


class ProgressiveHead(nn.Module):
    def __init__(self,projector,text,lut,shared_projection=False,chunk=512,
                 force_full=False,capture_states=False,fused_software=False,query_proxy=False,parent_window=None,
                 deferred_output=False):
        super().__init__()
        if not isinstance(projector.output_norm,nn.Identity):
            raise ValueError('Channel replacement requires an unnormalized affine MLP output')
        self.enabled=True;self.shared_projection=shared_projection;self.chunk=chunk
        self.checkpoint_channels=channel_checkpoints(projector.mlp_head[0].out_features,chunk,
            getattr(lut,'config',{}).get('channel_checkpoint_growth',1))
        self.force_full=force_full;self.capture_states=capture_states
        self.fused_software=fused_software
        self.query_proxy=query_proxy
        self.deferred_output=deferred_output
        if deferred_output and not query_proxy:raise ValueError('Deferred output requires the query proxy')
        self.parent_window=parent_window
        if parent_window is not None and parent_window<1:raise ValueError('Parent window must be positive')
        if query_proxy and not fused_software:raise ValueError('Query proxy requires the fused execution path')
        self.norm=projector.input_norm
        first,last=projector.mlp_head[0],projector.mlp_head[2]
        if hasattr(lut,'checkpoint_channels') and not np.array_equal(
                lut.checkpoint_channels,self.checkpoint_channels):
            raise ValueError('Calibration and actual hidden-channel domains differ')
        self.query_transposed=lut.config.get('channel_query_transposed',False)
        query=((text@last.weight.float()).T if self.query_transposed else last.weight.float().T@text.T)
        order=(first.weight.float().norm(dim=1)*query.norm(dim=1)).argsort(descending=True)
        self.register_buffer('first_weight',first.weight[order].detach())
        self.register_buffer('first_bias',first.bias[order].detach())
        self.register_buffer('second_weight',last.weight[:,order].detach())
        self.register_buffer('second_bias',last.bias.detach())
        self.register_buffer('texts',text)
        if query_proxy:self.register_buffer('query_weight',query[order].T.half().contiguous())
        for depth,((a,b),table) in enumerate(zip(lut.edges,lut.table)):
            self.register_buffer(f'tv_edge_{depth}',torch.as_tensor(a,device=text.device,dtype=torch.float32))
            self.register_buffer(f'margin_edge_{depth}',torch.as_tensor(b,device=text.device,dtype=torch.float32))
            self.register_buffer(f'demand_{depth}',torch.as_tensor(table,device=text.device,dtype=torch.float32))
        self.window=lut.config['predictor_window'];self.bins=lut.config['predictor_bins']
        self.tolerance=min(lut.config['semantic_tolerance'],lut.config['class_repair_tolerance'])
        self.response_horizon=lut.config.get('correction_horizon','suffix')
        self.state_fp16=lut.config.get('channel_state_fp16',False)
        self.center_consumers=lut.config.get('channel_center_consumers',False)
        self.center_fused_distance=lut.config.get('channel_center_fused_distance',False)
        self.indexed_linear=lut.config.get('channel_indexed_linear',False)
        self.context_level=context_level(lut.config.get('channel_context_level',2))
        if self.indexed_linear and shared_projection:
            raise ValueError('Indexed affine requires the normalized input bank')
        if self.center_consumers and shared_projection:
            raise ValueError('Center selection requires the actual normalized input bank')
        if self.response_horizon not in ('suffix','final'):raise ValueError(self.response_horizon)
        self.depths=len(self.checkpoint_channels)
        if len(lut.table)!=self.depths:raise ValueError('Calibration/checkpoint count mismatch')
        if shared_projection:
            from chorus_runtime import CHANNELS
            self.affine=AncestorAffine(projector,CHANNELS,torch.float16,order)

    def hidden(self,state,ids,begin,end):
        if self.shared_projection:
            from .ancestor_triton import merge_gelu
            projections,statistics=state
            return merge_gelu(self.affine,projections,statistics,0,len(ids),True,
                              columns=(begin,end),selected=ids)
        if self.indexed_linear:
            from .progressive_triton import indexed_linear
            return F.gelu(indexed_linear(state,self.first_weight[begin:end],self.first_bias[begin:end],ids))
        return F.gelu(mapped_linear(state,self.first_weight[begin:end],self.first_bias[begin:end],ids))

    def semantic_score(self,value):
        # Every checkpoint would emit FP16. Include that real output boundary
        # in calibration, completed feedback and online stop observations.
        return F.normalize(value.half().float()@self.texts.T,dim=1)

    def materialize(self,replacements,selected,end,seed,inverse,output):
        """One real terminal projection of the processed hidden prefix.

        Unprocessed channels keep the completed seed contribution. Values
        beyond the processed prefix are never read. This new FP16 product
        boundary needs its own full-refinement capture and calibration.
        """
        if not len(selected):return
        value=seed[inverse[selected]].float()+mapped_linear(replacements[:,:end],
            self.second_weight[:,:end],None,selected).float()
        output[selected]=value.to(output.dtype)

    def advance(self,state,selected,begin,end,hidden,inverse,output,query_state,
                replacements=None,seed=None):
        current=self.hidden(state,selected,begin,end)
        if self.fused_software:
            from .progressive_triton import difference,update
            delta=difference(current,hidden,inverse,selected,begin)
            if self.deferred_output:
                replacements[selected,begin:end]=delta
                raw=update(query_state,F.linear(delta,self.query_weight[:,begin:end]),selected,half_observation=False)
                if end==len(self.first_bias):
                    self.materialize(replacements,selected,end,seed,inverse,output)
                    return output[selected].half().float()@self.texts.T
                return raw
            semantic=update(output,F.linear(delta,self.second_weight[:,begin:end]),selected,
                            materialize=not self.query_proxy)
            if self.query_proxy:
                raw=update(query_state,F.linear(delta,self.query_weight[:,begin:end]),selected,half_observation=False)
                # The completed-consumer terminal must observe the real
                # emitted output, including every FP16 arithmetic boundary.
                if end==len(self.first_bias):return output[selected].half().float()@self.texts.T
                return raw
            return semantic@self.texts.T
        difference=(current.float()-hidden[inverse[selected],begin:end].float()).half()
        output[selected]+=F.linear(difference,self.second_weight[:,begin:end]).float()
        return output[selected].half().float()@self.texts.T

    def forward(self,point):
        banks=point['_garda_context_banks'];mapping=banks[self.context_level][1]
        requested=point.get('_garda_output_demand')
        if requested is not None and (requested.dtype!=torch.bool or requested.shape!=mapping.shape
                                      or requested.device!=mapping.device):
            raise ValueError('Output demand must be a Boolean mask of the native row domain')
        groups=(len(banks[self.context_level][0]) if banks[self.context_level][0] is not None else int(mapping.max())+1)
        state=(self.affine.prepare(banks,inline_fine=True) if self.shared_projection
               else self.norm(point.feat))
        if self.parent_window is None:
            return self.run_window(state,mapping,groups,requested)
        # Each parent is owned by one finite window, so no completed feedback
        # crosses a window boundary. Projections remain normal cacheable data;
        # only seed hidden/output and live trajectories have bounded lifetime.
        result=torch.empty((len(mapping),len(self.second_bias)),device=mapping.device,dtype=torch.float16)
        executed=[0]*self.depths;pilots=0;responders=0;seed_rows=[];responder_rows=[]
        if self.capture_states:
            all_states=torch.empty((self.depths,len(mapping),len(self.texts)),device=mapping.device)
            if self.query_proxy and self.force_full:all_outputs=torch.empty_like(all_states)
        for begin in range(0,groups,self.parent_window):
            end=min(begin+self.parent_window,groups)
            selected=((mapping>=begin)&(mapping<end)).nonzero().flatten()
            if not len(selected):continue
            if self.shared_projection:
                projections,statistics=state
                local_state=(projections,[(mu,var,index[selected]) for mu,var,index in statistics])
            else:local_state=state[selected]
            output=self.run_window(local_state,mapping[selected]-begin,end-begin,
                                   None if requested is None else requested[selected])
            if self.capture_states and self.center_consumers:
                seed_rows.append(selected[self.seed_rows]);responder_rows.append(selected[self.responder_rows])
            result[selected]=output
            for depth,count in enumerate(self.executed_rows):executed[depth]+=count
            pilots+=self.pilots;responders+=self.feedback_rows
            if self.capture_states:all_states[:,selected]=self.states
            if self.capture_states and self.query_proxy and self.force_full:
                all_outputs[:,selected]=self.output_states
        self.executed_rows=executed;self.pilots=pilots;self.feedback_rows=responders
        if self.capture_states and self.center_consumers:
            self.seed_rows=torch.cat(seed_rows);self.responder_rows=torch.cat(responder_rows)
        if self.capture_states:self.states=all_states
        if self.capture_states and self.query_proxy and self.force_full:self.output_states=all_outputs
        return result

    def run_window(self,state,inverse,groups,requested=None):
        # Parent ownership stays active through the final output. Selection
        # uses either physical row order or the already-ready input bank.
        mapping=inverse
        first,responder=select_consumers(state,mapping,groups,self.center_consumers,self.center_fused_distance)
        self.seed_rows=first;self.responder_rows=responder
        hidden=self.hidden(state,first,0,len(self.first_bias))
        seed=F.linear(hidden,self.second_weight,self.second_bias)
        output=seed[inverse].to(torch.float16 if self.state_fp16 else torch.float32)
        query_state=(seed.float()@self.texts.T)[inverse] if self.query_proxy else None
        replacements=(torch.empty((len(mapping),len(self.first_bias)),device=mapping.device,
                                  dtype=self.first_weight.dtype) if self.deferred_output else None)
        if requested is not None and (requested.dtype!=torch.bool or requested.shape!=mapping.shape
                                      or requested.device!=mapping.device):
            raise ValueError('Output demand must be a Boolean mask of the native row domain')
        # Earlier depth exits cancel output work, never the completed seeds
        # and responders still required by surviving consumers. Nonrequested
        # returned rows carry supplier/scratch values, not new final outputs.
        active=(torch.ones(len(mapping),device=mapping.device,dtype=torch.bool)
                if requested is None else requested.clone());active[first]=False
        active[responder]=False
        active=active.nonzero().flatten()
        previous=torch.empty((len(mapping),len(self.texts)),device=mapping.device)
        delta=torch.zeros((len(mapping),self.window-1),device=mapping.device)
        self.executed_rows=[];self.pilots=len(first)+len(responder)
        # A second completed consumer follows the same seed/delta arithmetic.
        # Its actual future correction is available before unfinished rows run.
        collect_outputs=self.capture_states and self.query_proxy and self.force_full
        response_scores=[];response_outputs=[]
        ends=self.checkpoint_channels.tolist()
        intervals=list(zip([0,*ends[:-1]],ends))
        for begin,end in intervals:
            with checkpoint_program():
                response_scores.append(F.normalize(self.advance(state,responder,begin,end,hidden,inverse,output,query_state,
                                                                replacements,seed),dim=1))
            if collect_outputs and not self.deferred_output:
                response_outputs.append(F.normalize(output[responder].half().float(),dim=1)@self.texts.T)
        response_scores=torch.stack(response_scores)
        winners=response_scores.argmax(2)
        response=torch.full((self.depths,groups),torch.inf,device=mapping.device)
        response_winner=torch.full((self.depths,groups),-1,device=mapping.device,dtype=torch.int64)
        for depth in range(self.depths):
            future=response_scores[depth:] if self.response_horizon=='suffix' else response_scores[-1:]
            kept=future.gather(2,winners[depth][None,:,None].expand(len(future),-1,1)).squeeze(2)
            response[depth,inverse[responder]]=(future.amax(2)-kept).amax(0)
            response_winner[depth,inverse[responder]]=winners[depth]
        self.feedback_rows=len(responder)
        captured=[];emitted=[]
        def emitted_snapshot(depth,end,active):
            # Labels for independent calibration only; never used online.
            if self.deferred_output:
                self.materialize(replacements,responder,end,seed,inverse,output)
                self.materialize(replacements,active,end,seed,inverse,output)
            snapshot=F.normalize(output.half().float(),dim=1)@self.texts.T
            if not self.deferred_output:snapshot[responder]=response_outputs[depth]
            return snapshot
        for depth,(begin,end) in enumerate(intervals):
            if not len(active):
                if not self.capture_states:break
                snapshot=(captured[-1].clone() if captured else
                          torch.empty((len(mapping),len(self.texts)),device=mapping.device))
                snapshot[first]=self.semantic_score(seed)
                snapshot[responder]=response_scores[depth]
                captured.append(snapshot);self.executed_rows.append(0)
                if collect_outputs:
                    emitted.append(emitted_snapshot(depth,end,active))
                continue
            self.executed_rows.append(len(active))
            # Both contributions originate from actually executed consumers.
            # FP16 products with FP32 running output are measured as a separate
            # numeric contract against the original complete model.
            if self.fused_software:from .progressive_triton import observe
            else:observe=observe_reference
            with checkpoint_program():
                raw=self.advance(state,active,begin,end,hidden,inverse,output,query_state,replacements,seed)
                score,keep=observe(raw,active,inverse,previous,delta,
                    getattr(self,f'tv_edge_{depth}'),getattr(self,f'margin_edge_{depth}'),getattr(self,f'demand_{depth}'),
                    response[depth],response_winner[depth],self.window,self.bins,depth,self.depths,self.tolerance,self.force_full)
            if self.capture_states:
                snapshot=(captured[-1].clone() if captured else
                          torch.empty((len(mapping),len(self.texts)),device=mapping.device))
                snapshot[first]=self.semantic_score(seed)
                snapshot[responder]=response_scores[depth]
                snapshot[active]=score;captured.append(snapshot)
                if collect_outputs:
                    emitted.append(emitted_snapshot(depth,end,active))
            if self.force_full or depth<self.window-1 or depth==self.depths-1:continue
            if self.deferred_output:
                self.materialize(replacements,active[~keep],end,seed,inverse,output)
            active=active[keep]
        if self.capture_states:self.states=torch.stack(captured)
        if collect_outputs:self.output_states=torch.stack(emitted)
        return output.half()


@contextmanager
def progressive_projector(model,text,lut,shared_projection=False,**options):
    projector=model.projectors.lang
    original_recovery=model.recovery.forward;original_projector=projector.forward
    projector.progressive=ProgressiveHead(projector,text,lut,shared_projection,**options).eval()
    def recover(self,point):
        if not projector.progressive.enabled:return original_recovery(point)
        point,banks=context_banks(point);point['_garda_context_banks']=banks
        if not shared_projection:point.feat=torch.cat([value[mapping] for value,mapping in banks],dim=1)
        return point
    def project(self,point):
        return self.progressive(point) if self.progressive.enabled else original_projector(point)
    model.recovery.forward=types.MethodType(recover,model.recovery)
    projector.forward=types.MethodType(project,projector)
    try:yield projector.progressive
    finally:
        model.recovery.forward=original_recovery;projector.forward=original_projector
        del projector.progressive


@torch.inference_mode()
def benchmark(dataset,destination,repeats=7,observations='results/raw/progressive_actual',fused=False,query_proxy=False,parent_window=None,policy=None,deferred_output=False,variant_names=None,channel_chunk=512):
    from statistics import median
    from chorus_runtime import texts
    from .numerical import metrics
    root=observation_root(observations,fused,query_proxy,deferred_output,channel_chunk,
                          bool(policy and policy.get('channel_state_fp16',False)),
                          bool(policy and policy.get('channel_query_transposed',False)),
                          bool(policy and policy.get('channel_center_consumers',False)),
                          bool(policy and policy.get('channel_center_fused_distance',False)),
                          bool(policy and policy.get('channel_indexed_linear',False)),
                          (policy or {}).get('channel_context_level',2),
                          (policy or {}).get('channel_checkpoint_growth',1))
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model,call,context,entry=load_case('Chorus',dataset,0);text=texts(dataset).cuda()
        report=dict(model='Chorus',dataset=dataset,entry=entry,variants={},
            scope='First real fragment; full 3D encoding; alternating RTX measurements',
            calibration='Both other datasets; no target labels in execution',
            status='Experimental checkpoint/readout refinement; no hardware pairing yet')
        with context:
            reference=call().clone()
            for variant in ('A','A_ancestor') if variant_names is None else variant_names:
                if variant not in ('A','A_ancestor'):raise ValueError(variant)
                b=variant=='A_ancestor'
                lut=calibrated_lut(dataset,variant,root,policy,channel_chunk)
                prepare_begin,prepare_end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                prepare_begin.record()
                with progressive_projector(model,text,lut,b,fused_software=fused,query_proxy=query_proxy,parent_window=parent_window,
                                           deferred_output=deferred_output,chunk=channel_chunk) as head:
                    prepare_end.record();prepare_end.synchronize()
                    result,baseline,samples=paired(call,head,repeats)
                    quality=metrics(result,reference,text)
                    counts=head.executed_rows;pilots=head.pilots;feedback=head.feedback_rows
                value=dict(quality=quality,baseline_samples=baseline,samples=samples,
                    measured_speedup=median(x['gpu_ms'] for x in baseline)/median(x['gpu_ms'] for x in samples),
                    pilots=pilots,completed_response_consumers=feedback,active_rows_per_checkpoint=counts,
                    model_query_prepare_gpu_ms=prepare_begin.elapsed_time(prepare_end))
                value['fused_software']=fused
                value['query_proxy']=query_proxy
                value['parent_window']=parent_window
                value['channel_chunk']=channel_chunk
                value['predictor_policy']=lut.config
                value['risk_calibration']=getattr(lut,'risk_calibration',None)
                value['deferred_output']=deferred_output
                report['variants'][variant]=value
                print(dataset,variant,value['measured_speedup'],quality,counts,flush=True)
        output=Path(destination);output.mkdir(parents=True,exist_ok=True)
        (output/f'Chorus_{dataset}_progressive_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


@torch.inference_mode()
def opportunity_observations(dataset,destination='results/raw/progressive'):
    """FP32 opportunity screen only; unsuitable for online FP16 calibration."""
    from chorus_runtime import texts
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model,call,context,entry=load_case('Chorus',dataset,0)
        text=texts(dataset).cuda();captured=[]
        handle=model.projectors.lang.mlp_head[1].register_forward_hook(lambda m,a,o:captured.append(o.detach()))
        try:
            with context:reference=call()
        finally:handle.remove()
        hidden=torch.cat(captured);first,last=model.projectors.lang.mlp_head[0],model.projectors.lang.mlp_head[2]
        query=last.weight.float().T@text.T
        order=(first.weight.float().norm(dim=1)*query.norm(dim=1)).argsort(descending=True)
        geometry=pickle.loads(Path(f'results/raw/Chorus_{dataset}_s0_maps.pkl').read_bytes())
        mapping=next(p['mapping'] for p in geometry['checkpoints'] if p['name'].endswith('enc2.block0'))
        unique,anchors,groups=np.unique(mapping,return_index=True,return_inverse=True)
        ids=torch.as_tensor(anchors,device='cuda');group=torch.as_tensor(groups,device='cuda')
        completed=hidden[ids]
        score=(completed.float()@query+last.bias.float()@text.T)[group]
        states=[];widths=[]
        for begin in range(0,len(order),512):
            selected=order[begin:begin+512]
            score+=(hidden[:,selected].float()-completed[:,selected][group].float())@query[selected]
            states.append(F.normalize(score,dim=1));widths.append(min(begin+512,len(order)))
        states=torch.stack(states);winner=states.argmax(2)
        final_score=reference.float()@text.T;target=final_score.argmax(1)
        top=states.topk(2,dim=2).values
        regret=final_score.max(1).values[None,:]-final_score.gather(1,winner.T).T
        arrays=dict(scores=states.cpu().numpy(),winner=winner.cpu().numpy(),target=target.cpu().numpy(),
            regret=regret.cpu().numpy(),margin=(top[:,:,0]-top[:,:,1]).cpu().numpy())
        root=Path(destination);root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(root/f'Chorus_{dataset}_progressive.npz',**arrays)
        report=dict(model='Chorus',dataset=dataset,entry=entry,hidden_channels=widths,
            pilot_fraction=len(anchors)/len(hidden),
            class_agreement=(winner==target[None,:]).float().mean(1).cpu().tolist(),
            within_tolerance=(regret<=.01).float().mean(1).cpu().tolist(),
            scope='Offline calibration/opportunity observations; not runtime or hardware speedup')
        (root/f'Chorus_{dataset}_progressive.json').write_text(json.dumps(report,indent=2)+'\n')
        print(dataset,report['class_agreement'],flush=True)
        return report
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def observation_root(directory,fused=False,query_proxy=False,deferred_output=False,channel_chunk=512,state_fp16=False,query_transposed=False,center_consumers=False,center_fused_distance=False,indexed_linear=False,shared_level=2,growth=1):
    if directory=='results/raw/progressive_actual':
        if deferred_output:directory='results/raw/progressive_deferred_actual'
        elif query_proxy:directory='results/raw/progressive_proxy_actual'
        elif fused:directory='results/raw/progressive_fused_actual'
    channel_checkpoints(1,channel_chunk,growth)
    root=Path(directory)/f'channels_{channel_chunk}' if channel_chunk!=512 else Path(directory)
    if state_fp16:root=root/'state_fp16'
    if query_transposed:root=root/'query_transposed'
    if center_consumers:root=root/'center_consumers'
    if center_fused_distance:root=root/'fused_distance'
    if indexed_linear:root=root/'indexed_linear'
    root=root/f'context_level_{shared_level}' if context_level(shared_level)!=2 else root
    return root/f'growth_{growth}' if growth!=1 else root


@torch.inference_mode()
def capture_observations(dataset,destination='results/raw/progressive_actual',fused=False,query_proxy=False,deferred_output=False,variant_names=None,channel_chunk=512,policy=None):
    """Calibrate the actual online FP16 arithmetic with all exits disabled.

    A and A_ancestor have separate rounding trajectories. Completed responders execute
    normally; no FP32 shortcut or target-row future enters a runtime decision.
    """
    from chorus_runtime import texts
    from .numerical import metrics
    lock=(WORKLOADS/'results/context_reload_v1/.gpu_counter.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        model,call,context,entry=load_case('Chorus',dataset,0)
        text=texts(dataset).cuda()
        config=channel_config(policy)
        root=observation_root(destination,fused,query_proxy,deferred_output,channel_chunk,
                              config.get('channel_state_fp16',False),config.get('channel_query_transposed',False),
                              config.get('channel_center_consumers',False),config.get('channel_center_fused_distance',False),
                              config.get('channel_indexed_linear',False),config.get('channel_context_level',2),config.get('channel_checkpoint_growth',1))
        root.mkdir(parents=True,exist_ok=True)
        channels=channel_checkpoints(model.projectors.lang.mlp_head[0].out_features,channel_chunk,
                                    config.get('channel_checkpoint_growth',1))
        depths=len(channels)
        stub=SimpleNamespace(config=config,edges=[(np.zeros(config['predictor_bins']-1),)*2]*depths,
                             table=[np.full(config['predictor_bins']**2,np.inf,np.float32)]*depths)
        with context:
            reference=call().clone()
            for variant in ('A','A_ancestor') if variant_names is None else variant_names:
                if variant not in ('A','A_ancestor'):raise ValueError(variant)
                b=variant=='A_ancestor'
                with progressive_projector(model,text,stub,b,force_full=True,capture_states=True,
                                           fused_software=fused,query_proxy=query_proxy,deferred_output=deferred_output,chunk=channel_chunk) as head:
                    result=call();states=head.states
                    quality=metrics(result,reference,text)
                    winner=states.argmax(2);top=states.topk(2,dim=2).values
                    final_score=reference.float()@text.T;target=final_score.argmax(1)
                    regret=final_score.max(1).values[None,:]-final_score.gather(1,winner.T).T
                    arrays=dict(scores=states.cpu().numpy(),winner=winner.cpu().numpy(),
                        target=target.cpu().numpy(),regret=regret.cpu().numpy(),
                        margin=(top[:,:,0]-top[:,:,1]).cpu().numpy(),checkpoint_channels=channels,
                        state_fp16=np.array(head.state_fp16),query_transposed=np.array(head.query_transposed),
                        center_consumers=np.array(head.center_consumers),center_fused_distance=np.array(head.center_fused_distance),
                        indexed_linear=np.array(head.indexed_linear),context_level=np.array(head.context_level))
                    if head.center_consumers:
                        arrays.update(seed_rows=head.seed_rows.cpu().numpy(),responder_rows=head.responder_rows.cpu().numpy())
                    if query_proxy:
                        output_winner=head.output_states.argmax(2)
                        target_score=F.normalize(reference.float(),dim=1)@text.T
                        output_regret=target_score.max(1).values[None,:]-target_score.gather(1,output_winner.T).T
                        arrays.update(output_winner=output_winner.cpu().numpy(),output_regret=output_regret.cpu().numpy())
                    report=dict(model='Chorus',dataset=dataset,variant=variant,entry=entry,
                        full_refinement_quality=quality,active_rows_per_checkpoint=head.executed_rows,
                        completed_consumers=head.pilots,response_consumers=head.feedback_rows,
                        fused_software=fused,
                        query_proxy=query_proxy,
                        deferred_output=deferred_output,
                        channel_chunk=channel_chunk,
                        state_fp16=head.state_fp16,
                        query_transposed=head.query_transposed,
                        center_consumers=head.center_consumers,
                        center_fused_distance=head.center_fused_distance,
                        indexed_linear=head.indexed_linear,
                        context_level=head.context_level,
                        class_agreement=(winner==target[None,:]).float().mean(1).cpu().tolist(),
                        within_tolerance=(regret<=.01).float().mean(1).cpu().tolist(),
                        arithmetic=f'Actual online FP16 blocks and {"FP16 stored" if head.state_fp16 else "FP32 running"} output; exits disabled',
                        scope='First real fragment; full 3D encoding; calibration data only')
                np.savez_compressed(root/f'Chorus_{dataset}_{variant}_progressive.npz',**arrays)
                (root/f'Chorus_{dataset}_{variant}_progressive.json').write_text(json.dumps(report,indent=2)+'\n')
                print(dataset,variant,'full refinement',quality,flush=True)
        return report
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
