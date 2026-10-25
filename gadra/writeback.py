"""Fold a sole adjacent vector cast into the original producer's writeback.

Logical dependencies stay native. No original storage is aliased to a different
precision: only the final cast allocation is physically written. Matrices,
reductions, mutable storage, observed intermediates and mismatched views fall
back. The vector unit still performs the native conversion after the producer.
"""
from collections import defaultdict
from .graph import key


def candidates(graph):
    # Diagnostic micrographs have their own physical programs and no captured
    # trace authority. Do not apply a native-tensor lowering to them.
    if not hasattr(graph,'trace'):return {}
    users=defaultdict(list);writers=defaultdict(list);versions=defaultdict(set)
    for at,event in enumerate(graph.events):
        for s in event['inputs']:users[key(s)].append(at);versions[s['storage']].add(s['version'])
        for s in event['outputs']:writers[key(s)].append(at);versions[s['storage']].add(s['version'])
    observed={key(s) for cp in graph.maps.get('checkpoints',[]) for s in
              ([cp['spec']] if 'spec' in cp else [])+cp.get('readout_inputs',[])}
    observed.update(key(s) for s in graph.trace.get('preparation_outputs',[]))
    pairs={}
    allowed=('aten.index.Tensor','aten.add.Tensor','aten.sub.Tensor','aten.mul.Tensor',
             'aten.div.Tensor','aten.relu.default','aten.gelu.default')
    for at,producer in enumerate(graph.events[:-1]):
        cast=graph.events[at+1]
        if (producer['kind']!='aten' or producer.get('op') not in allowed or
            len(producer['outputs'])!=1 or cast.get('op')!='aten.to.dtype' or
            len(cast['inputs'])!=1 or len(cast['outputs'])!=1):continue
        source=producer['outputs'][0];read=cast['inputs'][0];target=cast['outputs'][0]
        if (key(source)!=key(read) or key(source)==key(target) or
            users[key(source)]!=[at+1] or writers[key(source)]!=[at] or
            writers[key(target)]!=[at+1] or key(source) in observed or
            len(versions[source['storage']])!=1 or len(versions[target['storage']])!=1):continue
        if any(source[f]!=read[f] for f in ('address','shape','stride','element_bytes','dtype','bytes')):continue
        if (len(source['shape'])!=2 or source['shape']!=target['shape'] or
            any(s['role']!='feature' or not s['contiguous'] or
                s['stride']!=[s['shape'][1],1] or
                s['address']!=graph.storage[s['storage']]['address'] or
                s['bytes']!=graph.storage[s['storage']]['bytes'] for s in (source,target))):continue
        if (source.get('dtype')==target.get('dtype') or
            target['element_bytes']<source['element_bytes'] or
            key(target)==key(graph.events[-1]['outputs'][0])):continue
        pairs[at]=target
    return pairs


def join(sim,index,event,last_chunk,control):
    """Release the original cast descriptor only after its real target store."""
    ready=max(sim.cycle,sim.memory.clock,sim.stats['last_tile_completion'])
    sim.cycle=ready+sim.c['issue_cycles'];sim.memory.clock=sim.cycle
    sim.stats['writeback_cast_joins']+=1
    sim.stages[event['stage']]+=sim.c['issue_cycles']
    if sim.c.get('profile_events',False):
        sim.event_profile.append(dict(index=index,stage=event['stage'],kind='writeback_cast_join',
            op=event['op'],rows=0,shape=event['outputs'][0]['shape'],
            cycles=sim.c['issue_cycles'],memory=[0]*8,statistics=dict(writeback_cast_joins=1)))
    if index in sim.diagnostic_producers:sim.diagnostic_completions[index]=sim.cycle
    if last_chunk:sim.observe(index,control)
    sim.checkpoint_fork.join(sim,index,last_chunk)
