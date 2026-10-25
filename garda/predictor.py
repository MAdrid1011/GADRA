"""Causal TV/semantic-state lookup and already-completed consumer feedback."""
import numpy as np


def quality_error_budget(count,minimum):
    """Integer allowance using the same correct/count comparison as evaluation."""
    if count<=0 or not 0<=minimum<=1:
        raise ValueError('Quality budget requires positive count and a fraction in [0,1]')
    allowed=int(np.floor((1-minimum)*count))
    while allowed<count and (count-allowed-1)/count>=minimum:allowed+=1
    while (count-allowed)/count<minimum:allowed-=1
    return allowed


def exact_budget_selection(benefits, changes, budget):
    """Choose frozen LUT cells under integer calibration-scene budgets.

    This offline 0/1 program never receives inference observations. Runtime
    retains the ordinary finite table; negative error changes are permitted
    when an actual early readout corrects a full-depth fallback error.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    benefits=np.asarray(benefits,dtype=np.float64)
    changes=np.asarray(changes,dtype=np.int64)
    budget=np.asarray(budget,dtype=np.int64)
    if changes.shape!=(len(budget),len(benefits)):
        raise ValueError('Budget constraint and action dimensions differ')
    if not len(benefits):
        if np.any(budget<0):raise ValueError('Fallback already exceeds calibration budget')
        return np.empty(0,np.int64)
    solution=milp(-benefits,integrality=np.ones(len(benefits)),
        bounds=Bounds(0,1),constraints=LinearConstraint(changes,-np.inf,budget),
        options=dict(mip_rel_gap=0))
    if not solution.success:
        raise ValueError(f'Calibration budget optimization failed: {solution.message}')
    chosen=np.rint(solution.x).astype(np.int64)
    if np.any(chosen<0) or np.any(chosen>1) or np.any(changes@chosen>budget):
        raise AssertionError('Compiled integer actions violate calibration budgets')
    return np.flatnonzero(chosen)



def first_action_program(patterns, costs, benefits, budgets, cells):
    """Select cell bits with exactly the earliest enabled cell per pattern.

    `costs` has one matrix per pattern, rows by budget and columns by action.
    Continuous first-action variables are forced to 0/1 by binary cell bits;
    identical calibration cell sequences may share this action variable.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_array
    offsets = np.r_[cells, cells + np.cumsum([len(p) for p in patterns])]
    size = int(offsets[-1])
    if not size:
        if np.any(np.asarray(budgets) < 0):
            raise ValueError('Full-depth fallback exceeds calibration quality budget')
        return np.empty(0, np.int64), dict(benefit=0., spent=np.zeros(len(budgets), int).tolist(),
            allowance=np.asarray(budgets).tolist(), patterns=0, variables=0, binary_cells=0, constraints=0)
    objective = np.zeros(size)
    rr, cc, vv, lo, hi = [], [], [], [], []

    def constraint(entries, lower=-np.inf, upper=np.inf):
        row = len(lo)
        for column, value in entries:
            rr.append(row); cc.append(column); vv.append(value)
        lo.append(lower); hi.append(upper)

    for group, pattern in enumerate(patterns):
        for at, cell in enumerate(pattern):
            z = int(offsets[group] + at)
            objective[z] = -benefits[group][at]
            # z = x[cell] AND NOT(any earlier eligible x).
            constraint([(z, 1), (cell, -1)], upper=0)
            for earlier in pattern[:at]:
                constraint([(z, 1), (earlier, 1)], upper=1)
            constraint([(z, 1), (cell, -1)] + [(c, 1) for c in pattern[:at]], lower=0)
    for budget, allowance in enumerate(budgets):
        constraint([(int(offsets[g] + a), int(cost[budget, a]))
                    for g, cost in enumerate(costs) for a in range(cost.shape[1])
                    if cost[budget, a]], upper=int(allowance))
    matrix = coo_array((np.asarray(vv, float), (np.asarray(rr, np.int32), np.asarray(cc, np.int32))),
                       shape=(len(lo), size)).tocsc()
    integrality = np.r_[np.ones(cells), np.zeros(size-cells)]
    solution = milp(objective, integrality=integrality, bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, lo, hi), options=dict(mip_rel_gap=0))
    if not solution.success:
        raise ValueError(f'Joint calibration did not solve: {solution.message}')
    selected = np.rint(solution.x[:cells]).astype(bool)
    # Independently recompute deterministic first actions, costs and benefit.
    spent = np.zeros(len(budgets), np.int64); gain = 0.
    for group, pattern in enumerate(patterns):
        active = np.flatnonzero(selected[pattern])
        expected = np.zeros(len(pattern))
        if len(active):
            at = int(active[0]); expected[at] = 1
            spent += costs[group][:, at]; gain += benefits[group][at]
        np.testing.assert_allclose(solution.x[offsets[group]:offsets[group+1]], expected, atol=1e-6)
    if np.any(spent > budgets):
        raise AssertionError('Compiled first actions violate scene budgets')
    return np.flatnonzero(selected), dict(benefit=gain, spent=spent.tolist(),
        allowance=np.asarray(budgets).tolist(), patterns=len(patterns), variables=size,
        binary_cells=cells, constraints=len(lo))


def witness_layout(contexts,config):
    """Fixed owner slots selected by context/rank, independent of semantics."""
    _,group,counts=np.unique(contexts,return_inverse=True,return_counts=True)
    slots=config.get('witnesses_per_context',1)
    if slots<1 or config['witness_stride']<1:raise ValueError('Positive witness counts and stride required')
    owners=np.full((len(counts),slots),-1,np.int64)
    order=np.argsort(group,kind='stable');offset=np.r_[0,np.cumsum(counts)]
    for g in range(0,len(counts),config['witness_stride']):
        count=min(slots,counts[g])
        owners[g,:count]=order[offset[g]+np.arange(count)*counts[g]//count]
    return group,owners


def response_record_words(config):
    # A packed record stores a 15-bit semantic key, one valid bit and an
    # IEEE-FP16 repair value in one control-SRAM word. The unpacked format
    # keeps separate key, repair and valid words.
    if config.get('packed_response_records',False):
        if not config.get('response_match_winner',False):
            raise ValueError('Packed response records require semantic-state keys')
        return 1
    return 3 if config.get('response_match_winner',False) else 2


def uses_witness_selector(config):
    """Fixed owner selection must share allocation and execution barriers."""
    return (config.get('witnesses_per_context',1)>1 or
            config.get('response_match_winner',False) or
            config.get('predictor_conditioned_response',False))


def selector_storage_bytes(context_rows,leaf_rows,config):
    """Counts, ranks, retained owner IDs and leaf exclusion bitmap."""
    slots=config.get('witnesses_per_context',1)
    if not uses_witness_selector(config):return 0
    owner_bytes=selector_owner_bytes(leaf_rows)
    owner_words=(context_rows*slots*owner_bytes+3)//4
    return context_rows*8+owner_words*4+4*((leaf_rows+31)//32)


def selector_owner_bytes(leaf_rows):
    """Use 0xffff as the invalid 16-bit owner; larger domains use 32 bits."""
    return 2 if leaf_rows<=2**16-1 else 4


def selector_owner_addresses(contexts,slot_ids,context_rows,leaf_rows,config):
    """32-bit control-bank words containing invocation-static owner IDs."""
    slots=config.get('witnesses_per_context',1)
    owner_bytes=selector_owner_bytes(leaf_rows)
    per_word=4//owner_bytes
    contexts=np.asarray(contexts,dtype=np.int64)
    slot_ids=np.asarray(slot_ids,dtype=np.int64)
    return 2*context_rows+(contexts*slots+slot_ids)//per_word


def response_records(contexts,witnesses,winners,correction,config):
    """Install only completed owners into bounded per-context/depth slots."""
    group,owners=witness_layout(contexts,config)
    selected=np.flatnonzero(witnesses);expected=np.sort(owners[owners>=0])
    if not np.array_equal(selected,expected):raise ValueError('Completed producers do not own the response slots')
    positions=np.full(len(contexts),-1,np.int64);positions[selected]=np.arange(len(selected))
    valid=owners>=0
    keys=np.full((len(correction),*owners.shape),-1,np.int32)
    values=np.full(keys.shape,-np.inf,np.float32)
    keys[:,valid]=winners[:,positions[owners[valid]]]
    values[:,valid]=correction[:,positions[owners[valid]]]
    if config.get('packed_response_records',False):
        if np.any(keys[:,valid]>=2**15):
            raise ValueError('Packed response semantic key exceeds 15 bits')
        # Calibration and replay consume the exact value recovered from the
        # physical FP16 field, so packing cannot silently change quality.
        values=np.asarray(values,dtype=np.float16).astype(np.float32)
    return keys,values,group


def lookup_responses(records,current_winners,match=False):
    """Scan the finite owner set, max only applicable completed responses."""
    keys,values,group=records
    if not match:return values.max(2)[:,group]
    response=np.full(current_winners.shape,-np.inf,np.float32)
    for slot in range(values.shape[2]):
        response=np.maximum(response,np.where(current_winners==keys[:,:,slot][:,group],
                                               values[:,:,slot][:,group],-np.inf))
    return response


def feedback_gate_contexts(keys,values,response_edges,support,tolerance):
    """Conservative existential test over *completed* owner semantic keys.

    A context is rejected only when no possible current winner can reach an
    exit. Repeated keys use the ordinary max-response rule, never the most
    favorable owner. Missing keys cannot activate a conditional table cell.
    The unconditional prior, when usable, conservatively keeps every context.
    """
    prior,conditional=support
    if keys.shape!=values.shape or keys.ndim!=2:
        raise ValueError('Gate expects one checkpoint of finite owner records')
    if len(conditional)!=len(response_edges)+1:
        raise ValueError('Conditional support and response bins differ')
    possible=np.full(len(keys),bool(prior))
    for slot in range(keys.shape[1]):
        same=(keys==keys[:,slot,None])&(keys>=0)
        response=np.where(same,values,-np.inf).max(1)
        valid=(keys[:,slot]>=0)&np.isfinite(response)&(response<=tolerance)
        possible|=valid&np.asarray(conditional)[np.searchsorted(response_edges,response)]
    return possible


def query_scores(observations):
    score=observations.get('control_scores',observations['scores']).astype(np.float32)
    scale=np.maximum(np.linalg.norm(score,axis=2,keepdims=True),np.finfo(np.float32).tiny)
    return score/scale,scale


def select_checkpoints(observations,checkpoints,policy):
    if policy=='all':return observations,checkpoints
    if policy!='stage_end':raise ValueError(policy)
    indices=[i for i,cp in enumerate(checkpoints) if cp.get('stage_end') or cp['name']=='final']
    depth_count=len(checkpoints)
    selected={k:(v[indices] if v.ndim>=2 and v.shape[0]==depth_count else v) for k,v in observations.items()}
    selected['step_l1']=selected['stage_step_l1']
    return selected,[checkpoints[i] for i in indices]


def state_features(observations,window,space='language',history_fp16=False):
    if history_fp16 and space!='query_scores':
        raise ValueError('Packed query history requires query-score trajectories')
    if space=='language':
        delta=observations['step_l1'].astype(np.float32)
    elif space=='query_scores':
        # normalize(z @ text.T) == normalize(raw_readout @ text.T).
        # Thus a continue observation need not materialize the language vector.
        score,scale=query_scores(observations)
        delta=np.zeros(score.shape[:2],dtype=np.float32)
        previous=score[:-1].astype(np.float16).astype(np.float32) if history_fp16 else score[:-1]
        delta[1:]=np.abs(score[1:]-previous).sum(2)
    else:raise ValueError(space)
    tv=np.zeros_like(delta)
    for depth in range(len(delta)):
        begin=max(1,depth-window+2)
        if depth:tv[depth]=delta[begin:depth+1].sum(0)/max(1,depth-begin+1)
    margin=observations['margin'].astype(np.float32)
    if space=='query_scores':
        if 'control_scores' in observations:
            top=np.partition(score,-2,axis=2)[:,:,-2:]
            margin=top.max(2)-top.min(2)
        else:margin=margin/scale[:,:,0]
    return tv,margin


def demand_features(observations,config):
    """Use identical FP32 feature arithmetic for calibration and execution."""
    tv,margin=state_features(observations,config['predictor_window'],config.get('trajectory_space','language'),
                             config.get('diagnostic_history_fp16',False))
    if config.get('predictor_query_rms',False):
        if config.get('trajectory_space')!='query_scores':raise ValueError('Query RMS calibration requires query trajectories')
        scale=np.float32(np.sqrt(observations['scores'].shape[-1]))
        tv=tv*np.float32(1/scale)
        margin=margin*scale
    return tv,margin


def boundary_motion(observations,config,depth):
    """Sign of the current top-two gap change, using stored past scores.

    Current query IDs break all ties and select both previous components.
    No future stage or completed-owner label contributes this state bit.
    """
    if depth<1:raise ValueError('Boundary motion requires a previous observation')
    score=query_scores(observations)[0]
    current=score[depth]
    previous=score[depth-1].astype(np.float16).astype(np.float32)
    top=np.argsort(-current,axis=1,kind='stable')[:,:2]
    at=np.arange(len(current));winner,runner=top.T
    gap=current[at,winner]-current[at,runner]
    old_gap=previous[at,winner]-previous[at,runner]
    return (gap-old_gap>=np.float32(0)).astype(np.int64)


def response_state_keys(observations,config,depth,base):
    """Address the compressed sign planes; other stages keep their old key."""
    if config.get('predictor_margin_trend',False) and depth==config['predictor_first_exit']:
        return base+boundary_motion(observations,config,depth)*config['predictor_bins']**2
    return base


def completed_responses(observations,contexts,config):
    """Only deterministic pilot consumers contribute completed responses.

    Calibration invokes this with calibration labels; execution invokes it
    after the pilot phase. No unfinished consumer's future enters the map.
    """
    if config.get('response_selection','max')=='nearest':
        return nearest_completed_responses(observations,contexts,config)
    _,owners=witness_layout(contexts,config)
    witnesses=np.zeros(len(contexts),bool)
    witnesses[owners[owners>=0]]=True
    pilot={k:(v[:,witnesses] if v.ndim>=2 and v.shape[1]==len(contexts) else v)
           for k,v in observations.items()}
    correction=remaining_correction(pilot,config.get('correction_horizon','suffix'))
    records=response_records(contexts,witnesses,pilot['winner'],correction,config)
    match=config.get('response_match_winner',False)
    current=query_scores(observations)[0].argmax(2) if match else None
    return witnesses,lookup_responses(records,current,match)


def nearest_completed_responses(observations,contexts,config):
    """Select an applicable completed owner by current normalized scores."""
    group,owners=witness_layout(contexts,config)
    witnesses=np.zeros(len(contexts),bool);witnesses[owners[owners>=0]]=True
    selected=np.flatnonzero(witnesses)
    position=np.full(len(contexts),-1,np.int64);position[selected]=np.arange(len(selected))
    pilot={k:(v[:,witnesses] if v.ndim>=2 and v.shape[1]==len(contexts) else v)
           for k,v in observations.items()}
    correction=remaining_correction(pilot,config.get('correction_horizon','suffix'))
    score=query_scores(observations)[0]
    stored=query_scores(pilot)[0].astype(np.float16).astype(np.float32)
    winner=score.argmax(2);distance=np.full(winner.shape,np.inf,np.float32)
    response=np.full(winner.shape,-np.inf,np.float32)
    for slot in range(owners.shape[1]):
        ids=owners[group,slot];leaves=np.flatnonzero(ids>=0);producer=position[ids[leaves]]
        difference=np.abs(score[:,leaves]-stored[:,producer]).sum(2)
        take=(winner[:,leaves]==pilot['winner'][:,producer])&(difference<distance[:,leaves])
        response[:,leaves]=np.where(take,correction[:,producer],response[:,leaves])
        distance[:,leaves]=np.where(take,difference,distance[:,leaves])
    return witnesses,response


def joint_context_actions(keys,values,max_edges,nearest_edges,max_actions,nearest_actions,tolerance):
    """Necessary gate and score need for two or three fixed completed owners.

    Distance is useful only when matching owners disagree in at least one
    frozen state cell where their maximum response would continue. Distinct
    semantic keys must be reduced separately; an unrelated owner cannot
    suppress the need for distance between two same-key owners.
    """
    if keys.ndim!=2 or keys.shape[1] not in (2,3) or keys.shape!=values.shape:
        raise ValueError('Joint response requires two or three finite owner slots')
    support=(False,np.any(max_actions,axis=0))
    possible=feedback_gate_contexts(keys,values,max_edges,support,tolerance)
    flags=[]
    for slot in range(keys.shape[1]):
        valid=(keys[:,slot]>=0)&np.isfinite(values[:,slot])&(values[:,slot]<=tolerance)
        action=nearest_actions[:,np.searchsorted(nearest_edges,values[:,slot])].T&valid[:,None]
        flags.append(action);possible|=action.any(1)
    flags=np.stack(flags,axis=1)
    need_scores=np.zeros(len(keys),bool)
    for slot in range(keys.shape[1]):
        same=(keys==keys[:,slot,None])&(keys[:,slot,None]>=0)
        maximum=np.where(same,values,-np.inf).max(1)
        maximum_action=max_actions[:,np.searchsorted(max_edges,maximum)].T&(
            np.isfinite(maximum)&(maximum<=tolerance))[:,None]
        different=np.any(flags&same[:,:,None],axis=1)&np.any((~flags)&same[:,:,None],axis=1)
        need_scores|=np.any(different&~maximum_action,axis=1)
    return possible,need_scores


class DemandLUT:
    def feedback_gate_support(self):
        """Compile a necessary exit condition, without inference values.

        The first implementation masks the complete two-observation window
        of a single action depth. It does not approximate TV, change actions,
        or inspect an unfinished consumer's semantic trajectory.
        """
        c=self.config;depth=c.get('predictor_first_exit')
        if c.get('diagnostic_feedback_actions_only',False) and c.get('correction_horizon')!='final':
            raise ValueError('Action-only pilot feedback requires the final response horizon')
        if (depth is None or depth!=c.get('predictor_last_exit') or
                c['predictor_window']!=2 or not self.response_table or
                not c.get('response_match_winner',False) or
                not c.get('diagnostic_row_windows',False) or
                not c.get('finite_diagnostic_pipeline',False)):
            raise ValueError('Feedback gate requires one action, two observations and keyed finite feedback')
        tolerance=min(c['semantic_tolerance'],c['class_repair_tolerance'])
        prior=bool(np.any(self.table[depth]<=tolerance))
        conditional=np.any(self.response_table[depth].reshape(-1,c['predictor_response_bins'])<=tolerance,axis=0)
        return prior,conditional

    def observation_end(self):
        """Exclusive last useful nonterminal observation, from frozen actions.

        No position's future or chosen stop enters this compiler gate. Pilots
        still complete their full trajectory to provide terminal feedback.
        """
        threshold=min(self.config['semantic_tolerance'],
                      self.config.get('class_repair_tolerance',self.config['semantic_tolerance']))
        conditional=getattr(self,'response_table',[])
        for depth in range(len(self.table)-2,self.config['predictor_window']-2,-1):
            nearest=getattr(self,'nearest_table',[])
            if (np.any(self.table[depth]<=threshold) or (conditional and np.any(conditional[depth]<=threshold))
                or (nearest and np.any(nearest[depth]<=threshold))):
                return depth+1
        return self.observation_begin()

    def observation_begin(self):
        """First score needed by any action reachable in the calibrated LUT.

        This compiler gate reads only calibration constants. Retain the entire
        TV window before the earliest possible exit, even when those preceding
        checkpoints cannot themselves exit. Both independent and conditional
        tables are checked: a response cannot lower an estimate below its
        selected table entry, so gated actions remain unreachable.
        """
        threshold=min(self.config['semantic_tolerance'],
                      self.config.get('class_repair_tolerance',self.config['semantic_tolerance']))
        for depth in range(self.config['predictor_window']-1,len(self.table)-1):
            conditional=getattr(self,'response_table',[])
            nearest=getattr(self,'nearest_table',[])
            if (np.any(self.table[depth]<=threshold) or (conditional and np.any(conditional[depth]<=threshold))
                or (nearest and np.any(nearest[depth]<=threshold))):
                return depth-self.config['predictor_window']+1
        return len(self.table)-1

    def __init__(self,training,config,training_contexts=None):
        self.config=config
        if config.get('predictor_margin_trend_preserve_gate',False) and not config.get('predictor_margin_trend',False):
            raise ValueError('Preserving response support requires margin trend')
        if config.get('predictor_margin_trend',False) and not (
                config.get('predictor_joint_response',False) and
                config.get('trajectory_space')=='query_scores' and
                config.get('diagnostic_history_fp16',False)):
            raise ValueError('Margin trend requires joint query-score actions and stored FP16 history')
        if config.get('predictor_joint_response',False):
            self.compile_joint_response(training,training_contexts)
            return
        self.edges=[];self.table=[]
        features=[demand_features(a,config) for a in training]
        targets=[remaining_correction(a,config.get('correction_horizon','suffix')) for a in training]
        depth_count=len(features[0][0]);bins=config['predictor_bins']
        for depth in range(depth_count):
            tv=np.concatenate([a[0][depth] for a in features])
            margin=np.concatenate([a[1][depth] for a in features])
            # A calibration target, never read by inference for this position.
            # Difference in all query scores measures remaining semantic
            # correction; a shared score offset is removed per checkpoint.
            target=np.concatenate([a[depth] for a in targets])
            qs=np.arange(1,bins)/bins
            te=np.quantile(tv,qs);me=np.quantile(margin,qs)
            if config.get('predictor_query_rms',False) or config.get('predictor_edge_fp32',False):
                te=te.astype(np.float32);me=me.astype(np.float32)
            ids=np.searchsorted(te,tv)*bins+np.searchsorted(me,margin)
            table=np.full(bins*bins,np.inf,dtype=np.float32)
            for cell in range(bins*bins):
                values=target[ids==cell]
                if len(values)>=config['predictor_min_samples']:
                    table[cell]=np.quantile(values,config['predictor_quantile'],method='higher')
            if config.get('predictor_shared_support',False):
                # A cell dominated by one calibration scene does not provide
                # evidence for the other scene. Require support in each and
                # retain the largest observed response quantile.
                individual=[]
                for feature,label in zip(features,targets):
                    identity=np.searchsorted(te,feature[0][depth])*bins+np.searchsorted(me,feature[1][depth])
                    local=np.full(bins*bins,np.inf,dtype=np.float32)
                    for cell in range(bins*bins):
                        values=label[depth][identity==cell]
                        if len(values)>=config['predictor_min_samples']:
                            local[cell]=np.quantile(values,config['predictor_quantile'],method='higher')
                    individual.append(local)
                table=np.maximum.reduce(individual)
            self.edges.append((te,me));self.table.append(table)
        self.response_edges=[];self.response_table=[]
        if config.get('predictor_conditioned_response',False):
            if training_contexts is None or len(training_contexts)!=len(training):
                raise ValueError('Response calibration requires independent training context mappings')
            completed=[completed_responses(a,c,config) for a,c in zip(training,training_contexts)]
            response_bins=config['predictor_response_bins']
            for depth in range(depth_count):
                te,me=self.edges[depth]
                active=[np.isfinite(r[depth])&(~w) for w,r in completed]
                pooled=np.concatenate([r[depth,mask] for (_,r),mask in zip(completed,active)])
                re=np.quantile(pooled,np.arange(1,response_bins)/response_bins) if len(pooled) else np.zeros(response_bins-1)
                if config.get('predictor_query_rms',False) or config.get('predictor_edge_fp32',False):re=re.astype(np.float32)
                local_tables=[]
                for feature,label,(_,response),mask in zip(features,targets,completed,active):
                    ids=(np.searchsorted(te,feature[0][depth])*bins+np.searchsorted(me,feature[1][depth]))*response_bins+np.searchsorted(re,response[depth])
                    table=np.full(bins*bins*response_bins,np.inf,dtype=np.float32)
                    for cell in np.unique(ids[mask]):
                        values=label[depth,mask&(ids==cell)]
                        if len(values)>=config['predictor_min_samples']:
                            table[cell]=np.quantile(values,config['predictor_quantile'],method='higher')
                    local_tables.append(table)
                # Conditional evidence must have support in every independent
                # training scene; unseen states fall back to the old prior.
                self.response_edges.append(re)
                self.response_table.append(np.maximum.reduce(local_tables))
            if self.storage_bytes()>config['control_sram_bytes']:
                raise ValueError('Calibrated predictor exceeds the finite control SRAM')
        risk=config.get('predictor_max_class_error')
        if risk is not None:
            if not 0<=risk<=1:raise ValueError('Class-error frequency must be in [0,1]')
            # Compile risky/unsupported cells into the existing continue
            # action. This requires no extra runtime lookup or risk SRAM.
            # Labels belong only to independent calibration scenes; an
            # empirical frequency is not a target-scene quality guarantee.
            for depth,(te,me) in enumerate(self.edges):
                allowed=np.ones(bins*bins,bool)
                for observation,feature in zip(training,features):
                    cell=np.searchsorted(te,feature[0][depth])*bins+np.searchsorted(me,feature[1][depth])
                    winner=observation.get('output_winner',observation['winner'])[depth]
                    error=winner!=observation['target']
                    for at in range(bins*bins):
                        selected=cell==at
                        if (selected.sum()<config['predictor_min_samples'] or
                            error[selected].mean()>risk):allowed[at]=False
                self.table[depth][~allowed]=np.inf
                if self.response_table:
                    self.response_table[depth].reshape(bins*bins,-1)[~allowed]=np.inf
        if 'predictor_first_exit' in config:
            first=config['predictor_first_exit']
            if (not isinstance(first,int) or isinstance(first,bool) or
                    not 0<=first<depth_count):
                raise ValueError('First exit must name an existing checkpoint')
            # Compile disabled actions into the existing continue value.
            # This removes unreachable observation prefixes through the
            # original observation_begin gate, without a runtime selector.
            for depth in range(first):
                self.table[depth].fill(np.inf)
                if self.response_table:self.response_table[depth].fill(np.inf)
        if 'predictor_last_exit' in config:
            last=config['predictor_last_exit']
            if (not isinstance(last,int) or isinstance(last,bool) or
                    not 0<=last<depth_count-1 or last<config.get('predictor_first_exit',0)):
                raise ValueError('Last exit must name a nonterminal checkpoint at or after first exit')
            for depth in range(last+1,depth_count-1):
                self.table[depth].fill(np.inf)
                if self.response_table:self.response_table[depth].fill(np.inf)
        if config.get('predictor_conditional_error_budget',False):
            self.compile_conditional_error_budget(training,training_contexts)

    def compile_joint_response(self,training,contexts):
        """One calibration-scene budget for overlapping max/nearest actions."""
        c=self.config;depth=c.get('predictor_first_exit')
        if (contexts is None or depth!=c.get('predictor_last_exit') or depth is None or
            (not isinstance(c.get('witnesses_per_context'),int) or isinstance(c.get('witnesses_per_context'),bool) or c['witnesses_per_context']<1) or not c.get('response_match_winner',False) or
            not c.get('predictor_conditioned_response',False) or c.get('response_selection','max')!='max' or
            c.get('correction_horizon')!='final' or c['predictor_window']!=2):
            raise ValueError('Joint response requires a single final-horizon action and positive fixed matched owners')
        refined=c.get('predictor_margin_trend',False)
        base=dict(c,predictor_joint_response=False,predictor_conditional_error_budget=False,
                  predictor_margin_trend=False,predictor_margin_trend_preserve_gate=False)
        axes=[DemandLUT(training,dict(base,response_selection=selection),contexts)
              for selection in ('max','nearest')]
        self.__dict__.update(axes[0].__dict__);self.config=c
        for left,right in zip(axes[0].edges,axes[1].edges):
            for a,b in zip(left,right):np.testing.assert_array_equal(a,b)
        self.nearest_edges=axes[1].response_edges
        self.nearest_table=axes[1].response_table
        bins=c['predictor_bins'];rb=c['predictor_response_bins']
        tolerance=min(c['semantic_tolerance'],c['class_repair_tolerance'])
        support=None
        if c.get('predictor_margin_trend_preserve_gate',False):
            # Compile on the same independent calibration only. The incumbent
            # joint program defines permitted response bins, not target owners.
            incumbent=DemandLUT(training,dict(c,predictor_margin_trend=False,
                predictor_margin_trend_preserve_gate=False),contexts)
            support=[(tables[depth].reshape(-1,rb)<=tolerance).any(0)
                     for tables in (incumbent.response_table,incumbent.nearest_table)]
        scenes=[]
        for obs,mapping in zip(training,contexts):
            tv,margin=demand_features(obs,c);te,me=self.edges[depth]
            identity=(np.searchsorted(te,tv[depth])*bins+np.searchsorted(me,margin[depth]))*rb
            cells=[];valid=[]
            for selection,axis in zip(('max','nearest'),axes):
                pilots,response=completed_responses(obs,mapping,dict(base,response_selection=selection))
                cells.append(identity+np.searchsorted(axis.response_edges[depth],response[depth]))
                valid.append((~pilots)&np.isfinite(response[depth])&(response[depth]<=tolerance))
            cells=np.array(cells).T[:,:,None]
            valid=np.array(valid).T
            if support is not None:
                for axis in range(2):valid[:,axis]&=support[axis][cells[:,axis,0]%rb]
            if refined:
                band=boundary_motion(obs,c,depth)
                cells=np.concatenate((cells,(band[:,None,None]+1)*(bins*bins*rb)+cells),axis=2)
            scenes.append(dict(n=len(mapping),cell=cells,valid=valid,
                error=obs.get('output_winner',obs['winner'])!=obs['target'],
                regret=obs.get('output_regret',obs['regret'])>c['semantic_tolerance']))
        cells_per_plane=bins*bins*rb
        candidates=[(axis,cell) for axis in range(2) for cell in range(cells_per_plane*(3 if refined else 1))
            if all(np.count_nonzero(s['valid'][:,axis]&np.any(s['cell'][:,axis]==cell,axis=1))>=c['predictor_min_samples']
                   for s in scenes)]
        patterns=[];costs=[];benefits=[];budgets=[]
        for scene,s in enumerate(scenes):
            budgets.extend(quality_error_budget(s['n'],c[threshold])-int(s[name][-1].sum())
                for threshold,name in (('quality_min_class_agreement','error'),('quality_min_within_tolerance','regret')))
            sequence=np.full((s['n'],2*(2 if refined else 1)),-1,np.int32)
            for identity,(axis,cell) in enumerate(candidates):
                for kind in range(s['cell'].shape[2]):
                    sequence[s['valid'][:,axis]&(s['cell'][:,axis,kind]==cell),
                             axis*s['cell'].shape[2]+kind]=identity
            unique,group=np.unique(sequence,axis=0,return_inverse=True)
            for g,row in enumerate(unique):
                pattern=row[row>=0]
                if not len(pattern):continue
                mask=group==g;cost=np.zeros((2*len(scenes),len(pattern)),np.int64)
                for axis,name in enumerate(('error','regret')):
                    cost[2*scene+axis,:]=int(s[name][depth,mask].sum())-int(s[name][-1,mask].sum())
                patterns.append(pattern);costs.append(cost)
                benefits.append(np.full(len(pattern),mask.sum()/s['n']/len(scenes)))
        chosen,report=first_action_program(patterns,costs,benefits,np.array(budgets),len(candidates))
        if refined:
            # Original unsplit actions apply in both sign planes. Collapse
            # that union offline so runtime still reads one cell per axis.
            self.response_table[depth]=np.full(2*cells_per_plane,np.inf,np.float32)
            self.nearest_table[depth]=np.full(2*cells_per_plane,np.inf,np.float32)
        for table in self.table+self.response_table+self.nearest_table:table.fill(np.inf)
        report['actions']=[candidates[i] for i in chosen]
        for axis,cell in report['actions']:
            table=(self.response_table if axis==0 else self.nearest_table)[depth]
            if refined and cell<cells_per_plane:
                table[[cell,cell+cells_per_plane]]=tolerance
            else:table[cell-cells_per_plane if refined else cell]=tolerance
        self.joint_compilation=report
        if self.storage_bytes()>c['control_sram_bytes']:raise ValueError('Joint tables exceed control SRAM')

    def compile_conditional_error_budget(self,training,contexts):
        """Compile one checkpoint's existing cells using calibration-only risk.

        This is an empirical scene budget, not a distribution-free guarantee.
        Runtime still reads the same TV/margin/completed-response table and
        maxes the result with the actual applicable completed response.
        """
        config=self.config
        solver=config.get('predictor_conditional_budget_solver','greedy')
        if solver not in ('greedy','exact'):
            raise ValueError('Conditional budget solver must be greedy or exact')
        depth=config.get('predictor_first_exit')
        if (depth is None or depth!=config.get('predictor_last_exit') or
                not self.response_table or contexts is None):
            raise ValueError('Conditional error budget requires one exit checkpoint and completed-response tables')
        bins=config['predictor_bins'];rb=config['predictor_response_bins']
        tolerance=min(config['semantic_tolerance'],config['class_repair_tolerance'])
        te,me=self.edges[depth];scenes=[]
        for obs,mapping in zip(training,contexts):
            tv,margin=demand_features(obs,config)
            witnesses,response=completed_responses(obs,mapping,config)
            cell=(np.searchsorted(te,tv[depth])*bins+np.searchsorted(me,margin[depth]))*rb
            cell+=np.searchsorted(self.response_edges[depth],response[depth])
            winner=obs.get('output_winner',obs['winner'])
            regret=obs.get('output_regret',obs['regret'])
            final_error=(winner[-1]!=obs['target']).astype(np.int8)
            final_regret=(regret[-1]>config['semantic_tolerance']).astype(np.int8)
            scenes.append(dict(n=len(mapping),cell=cell,
                valid=(~witnesses)&np.isfinite(response[depth])&(response[depth]<=tolerance),
                final_error=final_error,final_regret=final_regret,
                error=(winner[depth]!=obs['target']).astype(np.int8)-final_error,
                regret=(regret[depth]>config['semantic_tolerance']).astype(np.int8)-final_regret))
        candidates=[]
        for cell in range(bins*bins*rb):
            masks=[s['valid']&(s['cell']==cell) for s in scenes]
            if all(mask.sum()>=config['predictor_min_samples'] for mask in masks):
                benefit=min(mask.sum()/s['n'] for s,mask in zip(scenes,masks))
                cost=max(max(s['error'][mask].sum(),s['regret'][mask].sum())/s['n']
                         for s,mask in zip(scenes,masks))
                candidates.append((benefit/max(cost,1/min(s['n'] for s in scenes)),cell,masks))
        errors=np.asarray([s['final_error'].sum() for s in scenes],np.int64)
        regrets=np.asarray([s['final_regret'].sum() for s in scenes],np.int64);selected=[]
        calibration_class=config.get('predictor_calibration_min_class_agreement',
                                     config['quality_min_class_agreement'])
        calibration_regret=config.get('predictor_calibration_min_within_tolerance',
                                      config['quality_min_within_tolerance'])
        budget=np.asarray([quality_error_budget(s['n'],threshold)-int(s[field].sum())
                           for s in scenes for threshold,field in (
                               (calibration_class,'final_error'),
                               (calibration_regret,'final_regret'))],np.int64)
        if solver=='exact':
            benefits=[np.mean([mask.sum()/s['n'] for s,mask in zip(scenes,masks)])
                      for _,_,masks in candidates]
            changes=np.asarray([[int(s[field][mask].sum()) for s,mask in zip(scenes,masks)
                                 for field in ('error','regret')] for _,_,masks in candidates],np.int64)
            changes=changes.reshape(len(candidates),2*len(scenes)).T
            selected=[candidates[i][1] for i in exact_budget_selection(benefits,changes,budget)]
        else:
            for _,cell,masks in sorted(candidates,key=lambda value:(-value[0],value[1])):
                next_error=errors+[int(s['error'][mask].sum()) for s,mask in zip(scenes,masks)]
                next_regret=regrets+[int(s['regret'][mask].sum()) for s,mask in zip(scenes,masks)]
                if all(e<=quality_error_budget(s['n'],calibration_class) and
                       r<=quality_error_budget(s['n'],calibration_regret)
                       for s,e,r in zip(scenes,next_error,next_regret)):
                    errors,regrets=next_error,next_regret;selected.append(cell)
        for table in self.table+self.response_table:table.fill(np.inf)
        self.response_table[depth][selected]=tolerance
        self.budget_selected_cells=selected

    def storage_bytes(self):
        """Actual frozen payload, including two RMS broadcast constants.

        Legacy non-RMS thresholds retain their original FP64 representation;
        new RMS programs freeze every threshold to FP32 before binning.
        """
        extra=getattr(self,'nearest_table',[])+getattr(self,'nearest_edges',[])
        return sum(a.nbytes for a in extra)+(8 if self.config.get('predictor_query_rms',False) else 0)+sum(a.nbytes for a in self.table+self.response_table)+sum(
            a.nbytes for pair in self.edges for a in pair)+sum(a.nbytes for a in self.response_edges)

    def decide(self,observations,contexts):
        config=self.config;tv,margin=demand_features(observations,config)
        depth_count,n=tv.shape;bins=config['predictor_bins']
        edges=self.edges
        stop=np.full(n,depth_count-1,dtype=np.int16)
        # A bounded pilot group completes first. All its full dependencies are
        # explicitly executed in phase 1; other positions never use own future.
        witnesses,responses=completed_responses(observations,contexts,config)
        nearest=(nearest_completed_responses(observations,contexts,config)[1]
                 if config.get('predictor_joint_response',False) else None)
        feedback_count=0
        conditioned_rows=0;lowered_rows=0
        for depth in range(config['predictor_window']-1,depth_count-1):
            te,me=edges[depth]
            ids=np.searchsorted(te,tv[depth])*bins+np.searchsorted(me,margin[depth])
            estimate=self.table[depth][ids].copy()
            # Witness max correction is accumulated only for completed pilots.
            # Max combines conservatively with the independent-scene prior.
            response=responses[depth]
            feedback_count+=len(np.unique(contexts[np.isfinite(response)]))
            prior=estimate.copy()
            if self.response_table:
                rb=config['predictor_response_bins']
                ids=response_state_keys(observations,config,depth,ids)*rb+np.searchsorted(self.response_edges[depth],response)
                conditional=self.response_table[depth][ids]
                usable=np.isfinite(response)&np.isfinite(conditional)
                estimate[usable]=conditional[usable]
                conditioned_rows+=int((usable&(~witnesses)&(stop==depth_count-1)).sum())
            estimate=np.maximum(estimate,response)
            lowered_rows+=int(((estimate<prior)&(~witnesses)&(stop==depth_count-1)).sum())
            tolerance=min(config['semantic_tolerance'],config.get('class_repair_tolerance',config['semantic_tolerance']))
            eligible=(stop==depth_count-1)&(~witnesses)&(estimate<=tolerance)
            if nearest is not None:
                base=np.searchsorted(te,tv[depth])*bins+np.searchsorted(me,margin[depth])
                base=response_state_keys(observations,config,depth,base)
                index=base*config['predictor_response_bins']+np.searchsorted(self.nearest_edges[depth],nearest[depth])
                allowed=(self.nearest_table[depth][index]<=tolerance)&np.isfinite(nearest[depth])&(nearest[depth]<=tolerance)
                eligible|=(stop==depth_count-1)&(~witnesses)&allowed
            stop[eligible]=depth
        selected=(np.arange(n),stop)
        def output_metric(name):
            return observations.get('output_'+name,observations[name])[stop,np.arange(n)]
        chosen=output_metric('winner');regret=output_metric('regret')
        metrics=dict(class_agreement=float((chosen==observations['target']).mean()),
                     mean_cosine=float(output_metric('cosine').mean()),
                     rmse=float(np.sqrt(output_metric('squared_error').mean())),
                     mean_final_score_regret=float(regret.mean()),
                     max_final_score_regret=float(regret.max()),
                     within_tolerance=float((regret<=config['semantic_tolerance']).mean()),
                     exited_fraction=float((stop<depth_count-1).mean()),
                     mean_completed_checkpoints=float((stop+1).mean()),
                     witnesses=int(witnesses.sum()),feedback_updates=feedback_count,
                     conditioned_observations=conditioned_rows,response_lowered_observations=lowered_rows,
                     stop_histogram=np.bincount(stop,minlength=depth_count).tolist())
        return stop,witnesses,metrics


def remaining_correction(observations,horizon='suffix'):
    """Calibration/finished-pilot label: future advantage over current winner.

    The query scores are normalized language cosines. This measures a repair
    that can change the semantic decision, rather than a harmless common score
    translation. The optional final horizon calibrates the output endpoint,
    while suffix also penalizes every transient intermediate winner. Inference
    never evaluates either label for unfinished consumers.
    """
    score=observations['scores'].astype(np.float32)
    winner=observations['winner']
    peak=score.max(2)
    if horizon=='final':
        return peak[-1][None,:]-score[-1][np.arange(score.shape[1])[None,:],winner]
    if horizon!='suffix':raise ValueError(horizon)
    result=np.zeros(score.shape[:2],dtype=np.float32)
    for depth in range(len(score)):
        retained=np.take_along_axis(score[depth:],winner[depth][None,:,None],axis=2).squeeze(2)
        result[depth]=(peak[depth:]-retained).max(0)
    return result
