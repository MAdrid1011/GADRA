"""Place sole-consumer chunks directly in their concatenation allocation.

Only physical addresses change. Logical specs, versions, demand closure and
all producers remain native. This is static buffer planning, not a data cache.
"""
from collections import defaultdict
from copy import copy

from .graph import key
from .protocol import SLOT_BYTES,SECTOR_BYTES


def lower(graph,*,channels=False,row_chunks=True):
    """Return an isolated physical layout, rejecting views and mutable owners."""
    result=copy(graph);result.bases=dict(graph.bases)
    result.concat_owners={};result.concat_joins={};result.concat_copies={}
    result.physical_views=dict(getattr(graph,'physical_views',{}))
    specs=defaultdict(list);producers=defaultdict(list);users=defaultdict(list)
    for index,event in enumerate(graph.events):
        for source in event['inputs']:
            specs[source['storage']].append(source);users[key(source)].append(index)
        for target in event['outputs']:
            specs[target['storage']].append(target);producers[key(target)].append(index)
    observed={key(s) for cp in graph.maps.get('checkpoints',[])
              for s in ([cp['spec']] if 'spec' in cp else [])+cp.get('readout_inputs',[])}
    assigned=set()

    def whole(spec):
        shape=spec['shape'];storage=graph.storage[spec['storage']]
        return (spec['role']=='feature' and len(shape)==2 and shape[0]>0 and shape[1]>0
                and spec['contiguous'] and spec['stride']==[shape[1],1]
                and spec['address']==storage['address']
                and spec['bytes']==storage['bytes']==shape[0]*shape[1]*spec['element_bytes']
                and {key(s) for s in specs[spec['storage']]}=={key(spec)})

    def movable(source,index):
        if (not whole(source) or key(source) in observed or users[key(source)]!=[index]
            or len(producers[key(source)])!=1 or producers[key(source)][0]>=index):return False
        output=graph.events[producers[key(source)][0]]['outputs']
        return len(output)==1 and all(output[0][f]==source[f] for f in
                                     ('address','shape','stride','element_bytes','bytes'))

    for index,event in enumerate(graph.events):
        if event.get('op')!='aten.cat.default' or len(event['outputs'])!=1:continue
        target=event['outputs'][0];sources=event['inputs']
        if (not whole(target) or len(sources)<2
            or any(s['role']!='feature' or len(s['shape'])!=2 for s in sources)):continue
        identities={key(s) for s in sources+[target]}
        if (len(identities)!=len(sources)+1 or identities&assigned
            or key(target) in observed or producers[key(target)]!=[index]
            or any(at<=index for at in users[key(target)])):continue
        if any(s['element_bytes']!=target['element_bytes'] or s.get('dtype')!=target.get('dtype') for s in sources):continue
        if result.bases[key(target)]%SLOT_BYTES:continue
        if (row_chunks and sum(s['shape'][0] for s in sources)==target['shape'][0]
            and all(s['shape'][1]==target['shape'][1] and movable(s,index) for s in sources)):
            # Continuous row chunks keep disjoint directory slot edges.
            if any(s['bytes']%SLOT_BYTES for s in sources[:-1]):continue
            offset=0
            for source in sources:
                result.bases[key(source)]=result.bases[key(target)]+offset
                result.concat_owners[key(source)]=key(target);offset+=source['bytes']
            result.concat_joins[index]=target['bytes'];assigned.update(identities)
        elif (channels and sum(s['shape'][1] for s in sources)==target['shape'][1]
              and all(s['shape'][0]==target['shape'][0] for s in sources)):
            # Only real indexed expansions may change their output row pitch.
            # The shared fine bank retains its original physical allocation.
            movable_ids=[j for j,s in enumerate(sources) if movable(s,index)
                         and graph.events[producers[key(s)][0]].get('op')=='aten.index.Tensor']
            remaining=[j for j in range(len(sources)) if j not in movable_ids]
            element=target['element_bytes'];pitch=target['shape'][1]*element
            if not movable_ids or len(remaining)!=1 or pitch%SECTOR_BYTES:continue
            if any(s['shape'][1]*element%SECTOR_BYTES for s in sources):continue
            column=0
            for j,source in enumerate(sources):
                width=source['shape'][1]
                if j in movable_ids:
                    result.physical_views[key(source)]=(result.bases[key(target)]+column*element,width*element,pitch)
                    result.concat_owners[key(source)]=key(target)
                else:
                    destination=dict(target,shape=[target['shape'][0],width],stride=[target['shape'][1],1],
                                     address=target['address']+column*element,bytes=target['shape'][0]*width*element,contiguous=False)
                    result.concat_copies[index]=(source,destination)
                column+=width
            assigned.update(identities)
    return result


def join(sim,index,selected,control,last_chunk):
    """Reuse the existing commit horizon and issue path; no payload is copied."""
    event=sim.g.events[index]
    before=max(sim.cycle,sim.memory.clock,sim.stats['last_tile_completion'],
               int(max(sim.matrix)),sim.vector)
    # Each static slice descriptor uses the existing task issue port. Its
    # destination base already travels with the ordinary producer store.
    cost=len(event['inputs'])*sim.c['issue_cycles']
    sim.cycle=before+cost;sim.memory.clock=max(sim.memory.clock,sim.cycle)
    size=len(selected)*event['outputs'][0]['shape'][1]*event['outputs'][0]['element_bytes']
    sim.stats['row_concat_elided_bytes']+=size
    sim.stats['row_concat_join_cycles']+=cost
    sim.stats['row_concat_joins']+=1
    sim.stages[event['stage']]+=cost
    if sim.c.get('profile_events',False):
        sim.event_profile.append(dict(index=index,stage=event['stage'],kind='row_concat_join',
            op=event['op'],rows=len(selected),shape=event['outputs'][0]['shape'],
            cycles=cost,memory=[0]*8,statistics=dict(row_concat_elided_bytes=size,row_concat_join_cycles=cost)))
    if index in sim.diagnostic_producers:sim.diagnostic_completions[index]=sim.cycle
    if last_chunk:sim.observe(index,control)
