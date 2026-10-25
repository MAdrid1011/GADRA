"""Sole-consumer dense GELU plans using the existing finite output owner."""
from .graph import key


def plans(graph,config):
    if not config.get('dense_column_gelu',False):return {}
    if not config.get('dense_outer_rows',False):
        raise ValueError('Column GELU requires dense outer row windows')
    from .feeder import use_dense_outer
    checkpoints=graph.maps.get('checkpoints',[])
    observed={key(s) for cp in checkpoints for s in cp.get('readout_inputs',[])}
    boundaries={cp['after_event'] for cp in checkpoints}
    boundaries.update(graph.maps.get('diagnostic_control_barriers',{}))
    boundaries.update(i for barrier in graph.maps.get('diagnostic_control_barriers',{}).values()
                      for i in barrier.get('producers',[]))
    result={}
    for i,event in enumerate(graph.events[:-1]):
        if not use_dense_outer(config,event) or len(event['outputs'])!=1 or i in boundaries or i+1 in boundaries:continue
        following=graph.events[i+1];source=event['outputs'][0]
        if (following.get('op')!='aten.gelu.default' or len(following['inputs'])!=1
            or len(following['outputs'])!=1 or following['stage']!=event['stage']):continue
        target=following['outputs'][0];operand=following['inputs'][0]
        fields=('address','shape','stride','element_bytes')
        if (key(operand)!=key(source) or any(operand[f]!=source[f] for f in fields)
            or not source['contiguous'] or not target['contiguous']
            or source['shape']!=target['shape'] or source['element_bytes']!=target['element_bytes']
            or key(source) in observed):continue
        users=[j for j,e in enumerate(graph.events) for s in e['inputs'] if key(s)==key(source)]
        if users!=[i+1]:continue
        result[i]=following
    return result
