"""Fuse ancestor projection gathers, LayerNorm correction and native GELU."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _merge(P0,P1,P2,P3,P4,M0,M1,M2,M3,M4,U0,U1,U2,U3,U4,V0,V1,V2,V3,V4,
           CORRECTION,CONSTANT,FINE_WEIGHT,OUTPUT,N:tl.constexpr,H:tl.constexpr,
           C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,C4:tl.constexpr,
           EPS:tl.constexpr,ROWS:tl.constexpr,COLS:tl.constexpr,INLINE_FINE:tl.constexpr,FINE_K:tl.constexpr,
           SOURCE_H:tl.constexpr,COLUMN_BEGIN:tl.constexpr,PROJECTED_COLUMN_BEGIN:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    out_column=tl.program_id(1)*COLS+tl.arange(0,COLS)
    column=out_column+COLUMN_BEGIN
    source_column=column-PROJECTED_COLUMN_BEGIN
    valid=row<N
    i0=tl.load(M0+row,valid,0);i1=tl.load(M1+row,valid,0)
    i2=tl.load(M2+row,valid,0);i3=tl.load(M3+row,valid,0);i4=tl.load(M4+row,valid,0)
    u0=tl.load(U0+i0,valid,0);u1=tl.load(U1+i1,valid,0)
    u2=tl.load(U2+i2,valid,0);u3=tl.load(U3+i3,valid,0);u4=tl.load(U4+i4,valid,0)
    width=C0+C1+C2+C3+C4
    mean=(u0*C0+u1*C1+u2*C2+u3*C3+u4*C4)/width
    variance=(C0*(tl.load(V0+i0,valid,0)+(u0-mean)*(u0-mean))+
              C1*(tl.load(V1+i1,valid,0)+(u1-mean)*(u1-mean))+
              C2*(tl.load(V2+i2,valid,0)+(u2-mean)*(u2-mean))+
              C3*(tl.load(V3+i3,valid,0)+(u3-mean)*(u3-mean))+
              C4*(tl.load(V4+i4,valid,0)+(u4-mean)*(u4-mean)))/width
    mask=valid[:,None]&(out_column[None,:]<H)
    if INLINE_FINE:
        k=tl.arange(0,FINE_K)
        fine=tl.load(P0+i0[:,None]*C0+k[None,:],valid[:,None]&(k[None,:]<C0),0)
        weight=tl.load(FINE_WEIGHT+column[None,:]*C0+k[:,None],(out_column[None,:]<H)&(k[:,None]<C0),0)
        # Match the separate FP16 first-bank affine storage boundary.
        value=tl.dot(fine,weight).to(tl.float16).to(tl.float32)
    else:
        value=tl.load(P0+i0[:,None]*SOURCE_H+source_column[None,:],mask,0).to(tl.float32)
    value+=tl.load(P1+i1[:,None]*SOURCE_H+source_column[None,:],mask,0).to(tl.float32)
    value+=tl.load(P2+i2[:,None]*SOURCE_H+source_column[None,:],mask,0).to(tl.float32)
    value+=tl.load(P3+i3[:,None]*SOURCE_H+source_column[None,:],mask,0).to(tl.float32)
    value+=tl.load(P4+i4[:,None]*SOURCE_H+source_column[None,:],mask,0).to(tl.float32)
    correction=tl.load(CORRECTION+column,out_column<H,0)
    constant=tl.load(CONSTANT+column,out_column<H,0)
    value=(value-mean[:,None]*correction[None,:])*tl.rsqrt(variance[:,None]+EPS)+constant[None,:]
    # The original first affine stores FP16 before GELU. Preserve that
    # rounding boundary even though gathers and correction are fused.
    value=value.to(tl.float16).to(tl.float32)
    gelu=value*0.5*(1+libdevice.erf(value*0.7071067811865475244))
    tl.store(OUTPUT+row[:,None]*H+out_column[None,:],gelu,mask)


def merge_gelu(affine,projections,statistics,begin,end,inline_fine=False,columns=None,selected=None,resolved_maps=None,
               projection_columns=None):
    if len(projections)!=5:raise ValueError('Chorus has five ancestor banks')
    maps=resolved_maps if resolved_maps is not None else [item[2][begin:end] if selected is None else item[2][selected] for item in statistics]
    means=[item[0] for item in statistics];variances=[item[1] for item in statistics]
    rows=end-begin if selected is None else len(selected)
    first,last=(0,len(affine.constant)) if columns is None else columns
    projected_first,projected_last=(0,len(affine.constant)) if projection_columns is None else projection_columns
    if not 0<=projected_first<=first<last<=projected_last<=len(affine.constant):
        raise ValueError('Requested columns exceed the materialized ancestor projection stripe')
    projected=[p for index,p in enumerate(projections) if not (index==0 and inline_fine)]
    source_stride=projected[0].stride(0)
    if any(p.ndim!=2 or p.shape[1]!=projected_last-projected_first or p.stride(1)!=1 or
           p.stride(0)!=source_stride for p in projected):
        raise ValueError('Ancestor projections must have matching real column domains and row strides')
    hidden=last-first
    result=torch.empty((rows,hidden),device=projections[0].device,dtype=torch.float16)
    block_rows,block_cols=(16,128) if inline_fine else (8,256)
    _merge[(triton.cdiv(rows,block_rows),triton.cdiv(hidden,block_cols))](
        *projections,*maps,*means,*variances,affine.mean_correction,affine.constant,affine.partials[0].weight,result,
        rows,hidden,*affine.channels,affine.eps,block_rows,block_cols,inline_fine,max(16,triton.next_power_of_2(affine.channels[0])),
        source_stride,first,projected_first,
        num_warps=4 if inline_fine else 8,enable_fp_fusion=False)
    return result
