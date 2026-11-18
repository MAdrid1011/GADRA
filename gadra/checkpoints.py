"""Model-specific checkpoint selection for the manuscript protocol."""
import re
import numpy as np


def stage_indices(model, checkpoints):
    patterns={
        'SceneSplat': (r'(?:backbone\.)?enc\.enc[1-4]\.block\d+',
                       r'(?:backbone\.)?dec\.dec[0-2]\.block\d+'),
        'Chorus': (r'(?:backbone\.)?enc\.enc[0-4]\.block\d+',),
        'EmbodiedSplat': (r'unet3d\.block[1-8]\.\d+',),
    }
    expected={'SceneSplat':7,'Chorus':5,'EmbodiedSplat':8}
    ids=[i for i,c in enumerate(checkpoints) if c.get('stage_end') and
         any(re.fullmatch(p,c['name']) for p in patterns[model])]
    if len(ids)!=expected[model]:
        raise ValueError(f'{model} requires {expected[model]} stage checkpoints, found {len(ids)}')
    return ids


def select(model, observations, checkpoints):
    ids=stage_indices(model,checkpoints)
    final=[i for i,c in enumerate(checkpoints) if c['name']=='final']
    if len(final)!=1:raise ValueError('Exactly one final output checkpoint is required')
    ids+=final
    result={k:(v[ids] if v.ndim>=2 and v.shape[0]==len(checkpoints) else v)
            for k,v in observations.items()}
    return result,[checkpoints[i] for i in ids]
