"""Captured reduction axes, exact row suppliers, and finite vector work."""
import math


OPERATIONS={'aten.mean.dim':'mean','aten.sum.dim_IntList':'sum',
            'aten.var_mean.dim':'var_mean',
            'aten.amax.default':'max','aten.argmax.default':'argmax',
            'aten.topk.default':'topk','aten.linalg_vector_norm.default':'norm',
            'aten.max.default':'max','aten.max.dim':'argmax',
            'aten.min.default':'max','aten.min.dim':'argmax'}


def capture(event,args,kwargs):
    operation=OPERATIONS.get(event.get('op'))
    if operation is None:return
    shape=list(args[0].shape);rank=len(shape)
    if operation=='norm':
        order=args[1] if len(args)>1 else kwargs.get('ord',2)
        if order!=2:return
        axes=args[2] if len(args)>2 else kwargs.get('dim')
        keep=args[3] if len(args)>3 else kwargs.get('keepdim',False)
    elif operation=='var_mean':
        axes=args[1] if len(args)>1 else kwargs.get('dim')
        unbiased=args[2] if len(args)>2 else kwargs.get('unbiased',True)
        keep=args[3] if len(args)>3 else kwargs.get('keepdim',False)
        event['reduction_correction']=int(bool(unbiased))
    elif operation=='topk':
        axes=args[2] if len(args)>2 else kwargs.get('dim',-1)
        keep=True
        event['reduction_k']=int(args[1] if len(args)>1 else kwargs['k'])
    else:
        axes=args[1] if len(args)>1 else kwargs.get('dim')
        keep=args[2] if len(args)>2 else kwargs.get('keepdim',False)
    if isinstance(axes,int):axes=[axes]
    elif axes is None or not len(axes):axes=list(range(rank))
    axes=sorted({int(axis)%rank for axis in axes}) if rank else []
    event.update(reduction_axes=axes,reduction_keepdim=bool(keep),reduction_operation=operation)


def lower_row_statistics(event):
    """Give verified channel statistics scalar row views at identical bytes.

    Native var_mean(dim=-1, keepdim=False) returns two prefix-shaped tensors.
    The simulator's last dimension denotes channels, so add a unit channel
    only for this captured reduction. Keep native axes/keepdim metadata and
    all storage, addresses and element strides for aliases unchanged.
    """
    if event.get('reduction_operation')!='var_mean':return
    source=event['inputs'][0]['shape'];outputs=event['outputs']
    if (len(source)<2 or event.get('reduction_axes')!=[len(source)-1]
        or event.get('reduction_keepdim') is not False or len(outputs)!=2
        or any(s['shape']!=source[:-1] for s in outputs)):return
    for spec in outputs:
        spec.update(shape=spec['shape']+[1],stride=spec['stride']+[1])
    event['reduction_scalar_row_view']=True


def row_local(event):
    """Only a verified last-axis reduction retaining physical row layout."""
    inputs=event.get('inputs',[]);outputs=event.get('outputs',[])
    if not inputs or not outputs:return False
    source=inputs[0]['shape'];out=outputs[0]['shape']
    return (len(source)>=2 and len(source)==len(out)
        and event.get('reduction_axes')==[len(source)-1]
        and (event.get('reduction_keepdim') is True or event.get('reduction_scalar_row_view') is True)
        and source[:-1]==out[:-1])


def vector_cycles(event,count,lanes):
    """Lane operations plus dependent reduction-tree drain, no extra unit.

    Norm is square/add/sqrt; mean is add/divide. Top-k uses a merge tree
    with k compare/select slots per input node, charged for all k slots.
    """
    operation=event.get('reduction_operation')
    if operation not in OPERATIONS.values():return None
    source=event['inputs'][0]['shape'];axes=event['reduction_axes']
    width=math.prod(source[axis] for axis in axes)
    out=event['outputs'][0]['shape'];physical_rows=math.prod(out[:-1]) if out else 1
    outputs=math.prod(out)*count/max(1,physical_rows)
    if operation=='var_mean':
        # Preserve the existing finite eight-pass FP32 statistics microcode;
        # narrowing the supplier rows must not make reduction work disappear.
        from .protocol import NORM_PASSES
        return math.ceil(outputs*width/lanes)*NORM_PASSES
    if operation=='topk':outputs/=max(1,event['reduction_k'])
    if not width or not outputs:return 0
    if operation=='norm':work=2*width  # width squares, width-1 adds, sqrt.
    elif operation=='mean':work=width  # width-1 adds, divide.
    elif operation=='argmax':work=2*(width-1)
    elif operation=='topk':work=2*event['reduction_k']*(width-1)
    else:work=width-1
    drain=math.ceil(math.log2(width))
    if operation in ('argmax','topk'):drain*=2
    return math.ceil(outputs*work/lanes)+drain
