"""Fused memory operations for the causal progressive software reference.

These kernels preserve FP32 subtraction/addition and the explicit FP16 stores.
They do not choose exits or approximate matrix products. Center-distance tree
rounding may affect consumer order, so that policy has separate calibration.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _response_owner(CURRENT,PILOT,PEERS,POSITIONS,OUTPUT,C:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK);valid=column<C
    current=tl.load(CURRENT+row*C+column,valid,0)
    best=tl.full((),float('inf'),tl.float32)
    selected=tl.full((),-1,tl.int64);owner=tl.full((),0,tl.int64)
    for slot in range(S):
        peer=tl.load(PEERS+row*S+slot)
        position=tl.load(POSITIONS+row*S+slot)
        pilot=tl.load(PILOT+position*C+column,valid,0)
        distance=tl.sum(tl.where(valid,tl.abs(current-pilot),0))
        take=(peer>=0)&(distance<best)
        best=tl.where(take,distance,best)
        selected=tl.where(take,peer,selected);owner=tl.where(take,position,owner)
    tl.store(OUTPUT+row*2,selected);tl.store(OUTPUT+row*2+1,owner)


def response_owner(current,pilot,peers,positions):
    """Select the same nearest completed owner without wide difference slabs."""
    if (current.ndim!=2 or pilot.ndim!=2 or current.shape[1]!=pilot.shape[1]
        or peers.ndim!=2 or peers.shape[1]<1 or len(peers)!=len(current)
        or positions.shape!=peers.shape
        or any(not x.is_cuda or not x.is_contiguous() for x in (current,pilot,peers,positions))
        or current.dtype!=torch.float32 or pilot.dtype!=torch.float32
        or any(x.dtype!=torch.int64 for x in (peers,positions))):
        raise ValueError('Owner selection requires contiguous FP32 CUDA queries and int64 peer maps')
    result=torch.empty((len(current),2),device=current.device,dtype=torch.int64)
    if len(current):
        _response_owner[(len(current),)](current,pilot,peers,positions,result,current.shape[1],
            peers.shape[1],triton.next_power_of_2(current.shape[1]),enable_fp_fusion=False)
    return result


@triton.jit
def _response_gate(OLD,NOW,CORRECTED,PREFIX,PILOT,AT,PEERS,POSITIONS,WINNER,
                   E0,E1,E2,E3,CHANGE,OUTPUT,C:tl.constexpr,S:tl.constexpr,
                   B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,B3:tl.constexpr,
                   MAGNITUDE:tl.constexpr,FOLDED:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK);valid=column<C
    old=tl.load(OLD+row*C+column,valid,0)
    now=tl.load(NOW+row*C+column,valid,0)
    at=tl.load(AT+row)
    if FOLDED:
        final_scores=tl.load(CORRECTED+at*C+column,valid,0)
        prefix=tl.load(PREFIX+at*C+column,valid,0)
        corrected=(now+final_scores)-prefix
    else:
        corrected=tl.load(CORRECTED+row*C+column,valid,0)
    old=old/tl.maximum(tl.sqrt(tl.sum(old*old)),1.e-12)
    now=now/tl.maximum(tl.sqrt(tl.sum(now*now)),1.e-12)
    new=corrected/tl.maximum(tl.sqrt(tl.sum(corrected*corrected)),1.e-12)
    maximum=tl.max(tl.where(valid,corrected,-float('inf')))
    winner=tl.min(tl.where(valid&(corrected==maximum),column,2147483647))
    first=tl.max(tl.where(valid,new,-float('inf')))
    second=tl.max(tl.where(valid&(column!=winner),new,-float('inf')))
    margin=first-second
    variation=tl.sum(tl.where(valid,tl.abs(now-old),0))
    pilot=tl.load(PILOT+at*C+column,valid,0)
    distance=tl.sum(tl.where(valid,tl.abs(now-pilot),0))
    agreement=tl.full((),0,tl.int64)
    for slot in range(S):
        peer=tl.load(PEERS+row*S+slot)
        producer=tl.load(POSITIONS+row*S+slot)
        final=tl.load(WINNER+producer)
        agreement+=((peer>=0)&(final==winner)).to(tl.int64)
    cell=agreement
    bucket=tl.full((),0,tl.int64)
    for edge in range(B0-1):bucket+=(tl.load(E0+edge)<variation).to(tl.int64)
    cell=cell*B0+bucket
    bucket=tl.full((),0,tl.int64)
    for edge in range(B1-1):bucket+=(tl.load(E1+edge)<margin).to(tl.int64)
    cell=cell*B1+bucket
    bucket=tl.full((),0,tl.int64)
    for edge in range(B2-1):bucket+=(tl.load(E2+edge)<distance).to(tl.int64)
    cell=cell*B2+bucket
    if MAGNITUDE:
        change=tl.load(CHANGE+at);bucket=tl.full((),0,tl.int64)
        for edge in range(B3-1):bucket+=(tl.load(E3+edge)<change).to(tl.int64)
        cell=cell*B3+bucket
    tl.store(OUTPUT+row,cell)


def response_gate(previous,current,corrected,pilot,at,peers,positions,winner,edges,change=None,*,folded_scores=None):
    """Actual normalized trajectory, agreement and frozen-LUT vector program."""
    rows,classes=current.shape
    folded=folded_scores is not None
    if folded:
        if corrected is not None or len(folded_scores)!=2:
            raise ValueError('Folded gate consumes completed final/prefix scores instead of corrected scores')
        final,prefix=folded_scores
        if final.shape!=pilot.shape or prefix.shape!=pilot.shape:
            raise ValueError('Folded scores must use the actual completed pilot domain')
        corrected=final
    else:prefix=pilot
    tensors=(previous,current,corrected,prefix,pilot,at,peers,positions,winner)+tuple(edges)
    if change is not None:tensors+=(change,)
    if (classes<2 or previous.shape!=current.shape or (not folded and corrected.shape!=current.shape)
        or pilot.ndim!=2 or pilot.shape[1]!=classes or at.shape!=(rows,)
        or peers.ndim!=2 or len(peers)!=rows or peers.shape[1]<1
        or positions.shape!=peers.shape or winner.shape!=(len(pilot),)
        or len(edges)!=3+int(change is not None) or any(e.ndim!=1 or not len(e) for e in edges)
        or any(not t.is_cuda or not t.is_contiguous() for t in tensors)
        or any(t.dtype!=torch.float32 for t in (previous,current,corrected,prefix,pilot)+tuple(edges))
        or any(t.dtype!=torch.int64 for t in (at,peers,positions,winner))
        or (change is not None and (change.shape!=(len(pilot),) or change.dtype!=torch.float32))):
        raise ValueError('Fused response gate requires contiguous real query, owner and LUT operands')
    result=torch.empty((rows,1),device=current.device,dtype=torch.int64)
    padded=tuple(edges)+(edges[-1],)*(4-len(edges))
    if rows:
        _response_gate[(rows,)](previous,current,corrected,prefix,pilot,at,peers,positions,winner,
            *padded,pilot if change is None else change,result,classes,peers.shape[1],
            *(len(e)+1 for e in padded),change is not None,folded,triton.next_power_of_2(classes),enable_fp_fusion=False)
    return result


@triton.jit
def _response_refine(LANGUAGE,FINAL,PREFIX,MAPPING,OUTPUT,W:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK);valid=column<W
    owner=tl.load(MAPPING+row)
    current=tl.load(LANGUAGE+row*W+column,valid,0)
    final=tl.load(FINAL+owner*W+column,valid,0)
    prefix=tl.load(PREFIX+owner*W+column,valid,0)
    # Preserve the original two FP32 rounding boundaries. Only the gathered
    # and arithmetic temporaries disappear; no gate or owner changes here.
    value=(current+final)-prefix
    norm=tl.maximum(tl.sqrt(tl.sum(value*value)),1.e-12)
    tl.store(OUTPUT+row*W+column,value/norm,valid)


def response_refine(language,final,prefix,mapping):
    """Mapped FP32 residual and row normalization in one real vector kernel."""
    if (language.ndim!=2 or final.ndim!=2 or prefix.shape!=final.shape
        or final.shape[1]!=language.shape[1] or mapping.shape!=(len(language),)
        or any(not x.is_cuda or not x.is_contiguous() for x in (language,final,prefix,mapping))
        or any(x.dtype!=torch.float32 for x in (language,final,prefix))
        or mapping.dtype not in (torch.int32,torch.int64)):
        raise ValueError('Response refinement requires contiguous CUDA FP32 rows and integer owners')
    result=torch.empty_like(language)
    if len(language):
        _response_refine[(len(language),)](language,final,prefix,mapping,result,language.shape[1],
            triton.next_power_of_2(language.shape[1]),enable_fp_fusion=False)
    return result


@triton.jit
def _affine_bank(P,E,M,I,WIDTH:tl.constexpr,row,column,valid,result,square,mean,C:tl.constexpr,CENTERED:tl.constexpr):
    actual=tl.load(I+row)
    result+=tl.load(P+actual*C+column,valid,0)
    square+=tl.load(E+actual)
    if CENTERED:mean+=tl.load(M+actual)*WIDTH
    return result,square,mean


@triton.jit
def _affine_center(M,K,I,WIDTH:tl.constexpr,row,column,valid,mean,result,square):
    actual=tl.load(I+row);delta=tl.load(M+actual)-mean
    result+=delta*tl.load(K+column,valid,0)
    square+=WIDTH*(delta*delta)
    return result,square


@triton.jit
def _merge_affine(P0,P1,P2,P3,P4,E0,E1,E2,E3,E4,M0,M1,M2,M3,M4,
                  K0,K1,K2,K3,K4,I0,I1,I2,I3,I4,BIAS,FINAL,PREFIX,OWNER,OUTPUT_IDS,OUT,
                  W0:tl.constexpr,W1:tl.constexpr,W2:tl.constexpr,W3:tl.constexpr,W4:tl.constexpr,
                  COUNT:tl.constexpr,TOTAL:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr,
                  CENTERED:tl.constexpr,DIVIDE:tl.constexpr,BIAS_STRIDE:tl.constexpr,
                  NORMALIZE:tl.constexpr,RESPONSE:tl.constexpr,SCATTER:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK);valid=column<C
    result=tl.full((BLOCK,),0,tl.float32)
    square=tl.full((),0,tl.float32);mean=tl.full((),0,tl.float32)
    if COUNT>0:
        result,square,mean=_affine_bank(P0,E0,M0,I0,W0,row,column,valid,result,square,mean,C,CENTERED)
    if COUNT>1:
        result,square,mean=_affine_bank(P1,E1,M1,I1,W1,row,column,valid,result,square,mean,C,CENTERED)
    if COUNT>2:
        result,square,mean=_affine_bank(P2,E2,M2,I2,W2,row,column,valid,result,square,mean,C,CENTERED)
    if COUNT>3:
        result,square,mean=_affine_bank(P3,E3,M3,I3,W3,row,column,valid,result,square,mean,C,CENTERED)
    if COUNT>4:
        result,square,mean=_affine_bank(P4,E4,M4,I4,W4,row,column,valid,result,square,mean,C,CENTERED)
    if CENTERED:
        mean=mean/TOTAL
        if COUNT>0:
            result,square=_affine_center(M0,K0,I0,W0,row,column,valid,mean,result,square)
        if COUNT>1:
            result,square=_affine_center(M1,K1,I1,W1,row,column,valid,mean,result,square)
        if COUNT>2:
            result,square=_affine_center(M2,K2,I2,W2,row,column,valid,mean,result,square)
        if COUNT>3:
            result,square=_affine_center(M3,K3,I3,W3,row,column,valid,mean,result,square)
        if COUNT>4:
            result,square=_affine_center(M4,K4,I4,W4,row,column,valid,mean,result,square)
    scale=tl.maximum(tl.sqrt(square),1.e-12)
    bias=tl.load(BIAS+column*BIAS_STRIDE,valid,0)
    value=result/scale+bias if DIVIDE else result+scale*bias
    if NORMALIZE:
        norm=tl.maximum(tl.sqrt(tl.sum(tl.where(valid,value*value,0))),1.e-12)
        value=value/norm
    if RESPONSE:
        owner=tl.load(OWNER+row)
        final=tl.load(FINAL+owner*C+column,valid,0)
        prefix=tl.load(PREFIX+owner*C+column,valid,0)
        value=(value+final)-prefix
        norm=tl.maximum(tl.sqrt(tl.sum(tl.where(valid,value*value,0))),1.e-12)
        value=value/norm
    target=tl.load(OUTPUT_IDS+row) if SCATTER else row
    tl.store(OUT+target*C+column,value,valid)


def merge_affine(terms,weight,divide_norm,centered,begin,end,normalize_output=False,*,response=None,destination=None):
    """Execute the native mapped-merge vector program without wide temporaries."""
    parts=tuple(t[0] for t in terms);energy=tuple(t[1] for t in terms)
    mean=tuple(t[2] for t in terms) if centered else energy
    coefficient=tuple(t[3] for t in terms) if centered else parts
    maps=tuple(t[5][begin:end] for t in terms);widths=tuple(t[4] for t in terms)
    tensors=parts+energy+mean+coefficient+maps
    if (not 1<=len(terms)<=5 or any(not t.is_cuda or not t.is_contiguous() for t in tensors)
        or any(t.dtype not in (torch.float16,torch.float32) for t in parts)
        or any(t.dtype!=torch.float32 for t in energy+mean+(coefficient if centered else ()))
        or any(t.dtype not in (torch.int32,torch.int64) for t in maps)):
        raise ValueError('Mapped affine merge requires contiguous CUDA half/FP32 partials, FP32 statistics and integer maps')
    if response is not None:
        final,prefix,owner=response
        if (not normalize_output or not divide_norm or final.ndim!=2 or prefix.shape!=final.shape
            or final.shape[1]!=weight.shape[1] or owner.shape!=(end-begin,)
            or any(not x.is_cuda or not x.is_contiguous() or x.device!=weight.device for x in response)
            or final.dtype!=torch.float32 or prefix.dtype!=torch.float32
            or owner.dtype not in (torch.int32,torch.int64)):
            raise ValueError('Mapped response requires normalized language, FP32 completed vectors and integer owners')
    else:final=prefix=owner=weight
    if destination is not None:
        result,output_ids=destination
        if (response is None or result.ndim!=2 or result.shape[1]!=weight.shape[1]
            or result.dtype!=torch.float32 or output_ids.shape!=(end-begin,)
            or output_ids.dtype not in (torch.int32,torch.int64)
            or any(not x.is_cuda or not x.is_contiguous() or x.device!=weight.device for x in destination)):
            raise ValueError('Mapped response scatter requires a real FP32 final buffer and selected row IDs')
    else:
        result=torch.empty((end-begin,weight.shape[1]),device=weight.device,dtype=torch.float32)
        output_ids=weight
    if end>begin:
        # Triton 2.x does not accept host tuples of pointers. The five native
        # hierarchy ports are explicit; unused ports are compile-time dead.
        pad=lambda values:values+(values[-1],)*(5-len(values))
        _merge_affine[(end-begin,)](*pad(parts),*pad(energy),*pad(mean),*pad(coefficient),*pad(maps),weight[-1],final,prefix,owner,output_ids,result,
            *pad(widths),len(widths),sum(widths),weight.shape[1],triton.next_power_of_2(weight.shape[1]),centered,divide_norm,weight.stride(1),
            normalize_output,response is not None,destination is not None,enable_fp_fusion=False)
    return result


@triton.jit
def _indexed_linear(STATE,WEIGHT,BIAS,IDS,RESULT,M:tl.constexpr,H:tl.constexpr,K:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    column=tl.program_id(1)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    actual=tl.load(IDS+row,row<M,0)
    accumulator=tl.zeros((BM,BN),tl.float32)
    for begin in range(tl.cdiv(K,BK)):
        k=begin*BK+inner
        value=tl.load(STATE+actual[:,None]*K+k[None,:],(row[:,None]<M)&(k[None,:]<K),0)
        weight=tl.load(WEIGHT+column[None,:]*K+k[:,None],(column[None,:]<H)&(k[:,None]<K),0)
        accumulator=tl.dot(value,weight,accumulator)
    bias=tl.load(BIAS+column,column<H,0).to(tl.float32)
    tl.store(RESULT+row[:,None]*H+column[None,:],accumulator+bias[None,:],
             (row[:,None]<M)&(column[None,:]<H))


def indexed_linear(state,weight,bias,ids):
    """Real half affine with indexed source loads, without a gathered matrix."""
    if (state.ndim!=2 or weight.ndim!=2 or state.shape[1]!=weight.shape[1]
        or bias.shape!=(len(weight),) or ids.ndim!=1
        or any(not x.is_contiguous() for x in (state,weight,bias,ids))
        or state.dtype!=torch.float16 or weight.dtype!=torch.float16 or bias.dtype!=torch.float16):
        raise ValueError('Indexed affine requires contiguous FP16 rows, weights, bias and row indices')
    m,h=len(ids),len(weight)
    result=torch.empty((m,h),device=state.device,dtype=torch.float16)
    _indexed_linear[(triton.cdiv(m,16),triton.cdiv(h,64))](state,weight,bias,ids,result,m,h,
        state.shape[1],16,64,32,enable_fp_fusion=False)
    return result


@triton.jit
def _center_distance(STATE,CENTER,MAPPING,DISTANCE,W:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK)
    parent=tl.load(MAPPING+row)
    value=tl.load(STATE+row*W+column,column<W,0).to(tl.float32)
    center=tl.load(CENTER+parent*W+column,column<W,0)
    difference=value-center
    square=difference*difference
    distance=tl.sqrt(tl.sum(tl.where(column<W,square,0)))
    tl.store(DISTANCE+row,distance)


def center_distance(state,center,mapping):
    """Read each ready row and its actual parent's center; store one scalar.

    FP32 arithmetic and an explicit square/reduction retain numerical meaning.
    The wide converted, gathered and difference matrices are never written.
    """
    if (state.ndim!=2 or center.ndim!=2 or state.shape[1]!=center.shape[1]
        or mapping.shape!=(len(state),) or not state.is_contiguous()
        or not center.is_contiguous() or not mapping.is_contiguous()):
        raise ValueError('Center distance requires contiguous rows and one parent per row')
    result=torch.empty(len(state),device=state.device,dtype=torch.float32)
    _center_distance[(len(state),)](state,center,mapping,result,state.shape[1],
        triton.next_power_of_2(state.shape[1]),enable_fp_fusion=False)
    return result


@triton.jit
def _difference(CURRENT,PILOT,MAPPING,IDS,RESULT,M:tl.constexpr,H:tl.constexpr,
                FULL_H:tl.constexpr,BEGIN:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=offset//H;column=offset%H;valid=row<M
    actual=tl.load(IDS+row,valid,0)
    parent=tl.load(MAPPING+actual,valid,0)
    current=tl.load(CURRENT+offset,valid,0).to(tl.float32)
    pilot=tl.load(PILOT+parent*FULL_H+BEGIN+column,valid,0).to(tl.float32)
    tl.store(RESULT+offset,current-pilot,valid)


@triton.jit
def _update(OUTPUT,CONTRIBUTION,IDS,SEMANTIC,M:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr,
            MATERIALIZE:tl.constexpr,HALF_OBSERVATION:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=offset//D;column=offset%D;valid=row<M
    actual=tl.load(IDS+row,valid,0)
    old=tl.load(OUTPUT+actual*D+column,valid,0)
    change=tl.load(CONTRIBUTION+offset,valid,0).to(tl.float32)
    value=old+change
    tl.store(OUTPUT+actual*D+column,value,valid)
    # Query observation must include the same half store as an emitted result.
    if MATERIALIZE:
        observation=value.to(tl.float16).to(tl.float32) if HALF_OBSERVATION else value
        tl.store(SEMANTIC+offset,observation,valid)


def difference(current,pilot,mapping,ids,begin):
    if not current.is_contiguous() or not pilot.is_contiguous():
        raise ValueError('Hidden rows must be contiguous')
    m,h=current.shape
    result=torch.empty_like(current)
    _difference[(triton.cdiv(m*h,256),)](current,pilot,mapping,ids,result,m,h,pilot.shape[1],begin,256)
    return result


def update(output,contribution,ids,materialize=True,half_observation=True):
    if not output.is_contiguous() or not contribution.is_contiguous():
        raise ValueError('Output rows must be contiguous')
    m,d=contribution.shape
    semantic=torch.empty((m,d),device=output.device,dtype=torch.float32) if materialize else None
    _update[(triton.cdiv(m*d,256),)](output,contribution,ids,semantic,m,d,256,materialize,half_observation,enable_fp_fusion=False)
    return semantic


@triton.jit
def _observe(RAW,IDS,MAPPING,PREVIOUS,DELTA,TV_EDGE,MARGIN_EDGE,TABLE,
             RESPONSE,RESPONSE_WINNER,SCORE,KEEP,Q:tl.constexpr,WINDOW:tl.constexpr,
             BINS:tl.constexpr,DEPTH:tl.constexpr,DEPTHS:tl.constexpr,
             TOLERANCE:tl.constexpr,FORCE_FULL:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.arange(0,BLOCK)
    actual=tl.load(IDS+row);valid=column<Q
    raw=tl.load(RAW+row*Q+column,valid,0)
    norm=tl.maximum(tl.sqrt(tl.sum(raw*raw)),1.e-12)
    score=raw/norm
    tl.store(SCORE+row*Q+column,score,valid)
    if DEPTH>0:
        old=tl.load(PREVIOUS+actual*Q+column,valid,0)
        change=tl.sum(tl.where(valid,tl.abs(score-old),0))
        slot=(DEPTH-1)%(WINDOW-1)
        tl.store(DELTA+actual*(WINDOW-1)+slot,change)
    tl.store(PREVIOUS+actual*Q+column,score,valid)
    keep=tl.full((),True,tl.int1)
    if (DEPTH>=WINDOW-1 and DEPTH<DEPTHS-1) and not FORCE_FULL:
        tv=tl.full((),0,tl.float32)
        for history in tl.static_range(WINDOW-1):
            if history==(DEPTH-1)%(WINDOW-1):tv+=change
            else:tv+=tl.load(DELTA+actual*(WINDOW-1)+history)
        tv=tv/(WINDOW-1)
        maximum=tl.max(tl.where(valid,score,-float('inf')))
        winner=tl.min(tl.where(valid&(score==maximum),column,2147483647))
        second=tl.max(tl.where(valid&(column!=winner),score,-float('inf')))
        margin=maximum-second
        edge=tl.arange(0,triton.next_power_of_2(BINS-1))
        a=tl.load(TV_EDGE+edge,edge<BINS-1,float('inf'))
        b=tl.load(MARGIN_EDGE+edge,edge<BINS-1,float('inf'))
        cell=tl.sum((tv>a).to(tl.int32))*BINS+tl.sum((margin>b).to(tl.int32))
        parent=tl.load(MAPPING+actual)
        applicable=tl.load(RESPONSE_WINNER+parent)==winner
        feedback=tl.where(applicable,tl.load(RESPONSE+parent),float('inf'))
        estimate=tl.maximum(tl.load(TABLE+cell),feedback)
        keep=estimate>TOLERANCE
    tl.store(KEEP+row,keep)


def observe(raw,ids,mapping,previous,delta,tv_edge,margin_edge,table,response,response_winner,
            window,bins,depth,depths,tolerance,force_full=False):
    score=torch.empty_like(raw);keep=torch.empty(len(ids),device=ids.device,dtype=torch.bool)
    _observe[(len(ids),)](raw,ids,mapping,previous,delta,tv_edge,margin_edge,table,
        response,response_winner,score,keep,raw.shape[1],window,bins,depth,depths,
        tolerance,force_full,triton.next_power_of_2(raw.shape[1]),enable_fp_fusion=False)
    return score,keep
