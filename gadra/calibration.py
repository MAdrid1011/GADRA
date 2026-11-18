"""Frozen quartile tables for remaining cumulative L1 semantic change."""
import numpy as np


def fit_tables(training, contexts, config):
    from .predictor import demand_features, remaining_correction, completed_responses
    if not training or contexts is None or len(training)!=len(contexts):
        raise ValueError('Calibration observations and their original context maps are required')
    features=[demand_features(o,config) for o in training]
    targets=[remaining_correction(o,'cumulative_l1') for o in training]
    feedback=[completed_responses(o,c,config) for o,c in zip(training,contexts)]
    depth_count=len(features[0][0]);bins=config['predictor_bins']
    if any(len(f[0])!=depth_count for f in features):
        raise ValueError('Calibration scenes must use identical checkpoint definitions')
    edges=[];tables=[];response_edges=[];response_tables=[]
    for depth in range(depth_count):
        tv=np.concatenate([f[0][depth] for f in features])
        margin=np.concatenate([f[1][depth] for f in features])
        target=np.concatenate([t[depth] for t in targets])
        te=np.quantile(tv,np.arange(1,bins)/bins).astype(np.float32)
        me=np.quantile(margin,np.arange(1,bins)/bins).astype(np.float32)
        local_ids=np.searchsorted(te,tv)*bins+np.searchsorted(me,margin)
        local=np.full(bins*bins,np.inf,np.float32)
        for cell in np.unique(local_ids):
            values=target[local_ids==cell]
            if len(values)>=config['predictor_min_samples']:
                local[cell]=np.quantile(values,.95,method='higher')
        edges.append((te,me));tables.append(local)
        valid=[(~owners)&np.isfinite(response[depth]) for owners,response in feedback]
        responses=np.concatenate([r[depth] for _,r in feedback])
        mask=np.concatenate(valid)
        re=(np.quantile(responses[mask],np.arange(1,bins)/bins)
            if mask.any() else np.zeros(bins-1)).astype(np.float32)
        ids=local_ids*bins+np.searchsorted(re,responses)
        table=np.full(bins**3,np.inf,np.float32)
        for cell in np.unique(ids[mask]):
            values=target[mask&(ids==cell)]
            if len(values)>=config['predictor_min_samples']:
                table[cell]=np.quantile(values,.95,method='higher')
        response_edges.append(re);response_tables.append(table)
    return edges,tables,response_edges,response_tables
