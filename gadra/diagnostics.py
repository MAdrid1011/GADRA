"""Exact readout/control sharing through existing hierarchical row maps."""
import numpy as np


def half_query(source,weight,text,bias=None,transposed=False):
    """FP32 query composition, FP16 matrix product, FP32 control state.

    The wide semantic readout keeps its original precision. This explicit
    control observation must be calibrated separately from FP32 observations.
    """
    import torch
    if transposed:
        augmented=weight.float() if bias is None else torch.cat((weight.float(),bias.float()[None]),dim=0)
        folded=(text.float()@augmented.T).T
        if bias is not None:source=torch.nn.functional.pad(source,(0,1),value=1)
    else:
        folded=weight.float()@text.float().T
    if bias is not None and not transposed:
        folded=torch.cat((folded,(bias.float()@text.float().T)[None]),dim=0)
        source=torch.nn.functional.pad(source,(0,1),value=1)
    return (source.half()@folded.half()).float()


def factorized_half_query(source,left,right,text):
    """Execute a frozen low-rank readout as two actual FP16 products."""
    if source.shape[1]!=left.shape[0] or left.shape[1]!=right.shape[0] or right.shape[1]!=text.shape[1]:
        raise ValueError('Factorized query dimensions do not compose')
    folded=right.float()@text.float().T
    latent=source.half()@left.half()
    return (latent@folded.half()).float()
from .graph import rows


def affine_merge_cost(classes,banks,centered,divide_norm):
    """Existing lane work and serial scalar/accumulation drain in cycles.

    The centered path first accumulates bank means, divides, then forms
    each delta/energy correction and the final sqrt, clamp and bias.
    Scalar operators use the model's ordinary one-cycle vector op contract.
    """
    operations=(classes*(4*banks+2+int(divide_norm))+8*banks+4 if centered
                else classes*(2*banks+2)+banks+2)
    dependency=3*banks+8 if centered else banks+4
    return operations,dependency


def native_index_payload(mapping,source_rows):
    """Functional result of the existing finite presence/rank microprogram.

    The packed int32 IDs and inverse share one physical allocation. Capture
    lowers this boundary to ``unique_index`` including its finite scratch,
    scans and conflicting stores; this is not a free host-side unique.
    """
    import torch
    if source_rows<0 or source_rows>=2**31:
        raise ValueError('Native source domain exceeds signed index width')
    used,inverse=torch.unique(mapping,sorted=True,return_inverse=True)
    payload=torch.cat((used,inverse)).to(torch.int32)[None,:]
    return payload,len(used)


class ReadoutNativePreparation:
    """Invocation-local supplies for immutable native checkpoint tensors.

    Only the same actual tensor object and centering mode share work. The
    caller keeps these stage-end sources immutable until all readouts finish.
    Values are ordinary captured tensors, never extra hardware storage.
    """
    def __init__(self):
        self.records=[]

    def get(self,source,centered):
        for original,mode,prepared in self.records:
            if original is source and mode==centered:return prepared
        values=source.float() if centered else source
        mean=values.mean(1,keepdim=True) if centered else None
        if centered:values=values-mean
        prepared=(values.half(),values.float().square().sum(1,keepdim=True),mean)
        self.records.append((source,centered,prepared))
        return prepared


def _partitioned_terms(banks,mappings,weight,centered,compact_native=False,native_preparation=None):
    """Canonical bank products; no leaf expansion or repeated matrix work."""
    import torch
    if len(banks)!=len(mappings) or not banks:raise ValueError('Native affine banks require matching maps')
    if sum(x.shape[1] for x in banks)+1!=weight.shape[0]:raise ValueError('Native affine bank width mismatch')
    terms=[];begin=0
    for values,mapping in zip(banks,mappings):
        mapping=torch.as_tensor(mapping,device=values.device,dtype=torch.long)
        if compact_native:
            payload,count=native_index_payload(mapping,len(values))
            values=values[payload[0,:count]]
            mapping=payload[0,count:]
        end=begin+values.shape[1];mean=coefficient=None
        if native_preparation is not None:
            if compact_native:raise ValueError('Shared native preparation requires the canonical source domain')
            prepared,square,mean=native_preparation.get(values,centered)
            if centered:coefficient=weight[begin:end].float().sum(0,keepdim=True)
            partial=prepared@weight[begin:end].half()
        elif centered:
            values=values.float();mean=values.mean(1,keepdim=True)
            coefficient=weight[begin:end].float().sum(0,keepdim=True)
            values=values-mean
        # Retain the actual FP16 product in its existing owner. Consumers
        # widen on load; a full canonical FP32 copy adds no information.
        if native_preparation is None:
            partial=values.half()@weight[begin:end].half()
            square=values.float().square().sum(1,keepdim=True)
        terms.append((partial,square,mean,coefficient,values.shape[1],mapping))
        begin=end
    return terms


def _merge_partitioned_terms(terms,weight,divide_norm,centered,begin,end):
    """Merge one fixed leaf interval, preserving the original sum order."""
    import torch
    result=torch.zeros((end-begin,weight.shape[1]),device=weight.device,dtype=torch.float32)
    square=torch.zeros((end-begin,1),device=weight.device,dtype=torch.float32)
    local=[]
    for partial,energy,mean,coefficient,width,mapping in terms:
        selected=mapping[begin:end]
        result.add_(partial[selected]);square.add_(energy[selected])
        if centered:local.append((mean[selected],width,coefficient))
    if centered:
        channels=sum(t[4] for t in terms)
        mean=sum(m*width for m,width,_ in local)/channels
        for m,width,coefficient in local:
            delta=m-mean
            result.add_(delta*coefficient);square.add_(width*delta.square())
    scale=square.sqrt().clamp_min(1e-12)
    return result/scale+weight[-1] if divide_norm else result+scale*weight[-1]


def partitioned_affine(banks,mappings,weight,*,divide_norm=False,centered=False,
                       row_window=None,normalize_output=False,merge_fused=False,
                       compact_native=False,native_preparation=None,fuse_normalization=False):
    """Project canonical banks once, merge bounded leaf windows through real maps.

    The optional row window bounds the lifetime of expanded partials and
    centered corrections. It adds no storage and changes neither suppliers
    nor matrix work. Every operation and window output is captured normally.
    ``normalize_output`` applies the ordinary row normalization before the
    window is released; changing its CUDA batch shape requires quality replay.
    """
    import torch
    from torch.nn import functional as F
    if row_window is not None and (not isinstance(row_window,int) or row_window<=0):
        raise ValueError('Positive integer affine row window required')
    if fuse_normalization and not (merge_fused and normalize_output):
        raise ValueError('Fused normalization requires the actual normalized merge kernel')
    terms=_partitioned_terms(banks,mappings,weight,centered,compact_native,native_preparation)
    count=len(mappings[0]);step=count if row_window is None else row_window
    if not count:
        return torch.empty((0,weight.shape[1]),device=weight.device,dtype=torch.float32)
    outputs=[]
    merge=_merge_partitioned_terms
    if merge_fused:
        from .progressive_triton import merge_affine
        merge=merge_affine
    for begin in range(0,count,step):
        args=(terms,weight,divide_norm,centered,begin,min(count,begin+step))
        value=merge(*args,normalize_output=True) if fuse_normalization else merge(*args)
        outputs.append(F.normalize(value,dim=1) if normalize_output and not fuse_normalization else value)
    return outputs[0] if len(outputs)==1 else torch.cat(outputs,dim=0)


def partitioned_language(banks,mappings,weight,*,centered=False,row_window=None,merge_fused=False,
                         compact_native=False,native_preparation=None,fuse_normalization=False):
    """Wide early output with the same native-row factorization as queries."""
    return partitioned_affine(banks,mappings,weight,divide_norm=True,centered=centered,
                              row_window=row_window,normalize_output=True,merge_fused=merge_fused,
                              compact_native=compact_native,native_preparation=native_preparation,
                              fuse_normalization=fuse_normalization)


class PartitionedLanguageWindows:
    """Real native supplies with only the requested leaf window materialized."""
    def __init__(self,banks,mappings,weight,*,centered=False,merge_fused=False):
        self.terms=_partitioned_terms(banks,mappings,weight,centered)
        self.weight=weight;self.centered=centered;self.merge_fused=merge_fused

    def evaluate(self,terms,begin,end):
        import torch.nn.functional as F
        merge=_merge_partitioned_terms
        if self.merge_fused:
            from .progressive_triton import merge_affine
            merge=merge_affine
        return F.normalize(merge(terms,self.weight,True,self.centered,begin,end),dim=1)

    def window(self,begin,end):
        return self.evaluate(self.terms,begin,end)

    def selected(self,ids):
        # Completed pilot prefixes use the same canonical bank products.
        # Actual mapping gathers remain tensor operations in the program.
        terms=[(*term[:5],term[5][ids]) for term in self.terms]
        return self.evaluate(terms,0,len(ids))


class SelectedLanguageReadout:
    """Build wide native products only for actual requested output maps."""
    def __init__(self,banks,mappings,weight,*,centered=False,merge_fused=False,fuse_normalization=False,
                 reuse_coarse_projections=False,direct_native_maps=False,direct_coarse_membership=False):
        if direct_native_maps and not reuse_coarse_projections:
            raise ValueError('Direct selected maps require retained native coarse products')
        if direct_coarse_membership and not direct_native_maps:
            raise ValueError('Direct coarse membership requires direct native maps')
        self.banks=banks;self.mappings=mappings;self.weight=weight
        self.centered=centered;self.merge_fused=merge_fused
        self.fuse_normalization=fuse_normalization
        self.reuse_coarse_projections=reuse_coarse_projections
        self.direct_native_maps=direct_native_maps
        self.direct_coarse_membership=direct_coarse_membership
        self.coarse_supplies={}

    def selected(self,ids):
        if self.reuse_coarse_projections:return self._selected_shared(ids)
        return partitioned_language(self.banks,[m[ids] for m in self.mappings],self.weight,
            centered=self.centered,merge_fused=self.merge_fused,compact_native=True,
            fuse_normalization=self.fuse_normalization)

    def refined_selected(self,ids,final,prefix,owners,*,output=None):
        """One real mapped vector program; no normalized-language slab."""
        if not (self.direct_native_maps and self.merge_fused and self.fuse_normalization):
            raise ValueError('Fused selected response requires direct normalized fused native maps')
        return self._selected_shared(ids,response=(final,prefix,owners),
            destination=None if output is None else (output,ids))

    def selected_windows(self,ids,row_window):
        """Fill actual coarse suppliers once, then release each leaf window."""
        if not self.direct_native_maps:
            raise ValueError('Selected windows require direct retained native products')
        if not isinstance(row_window,int) or row_window<1:
            raise ValueError('Positive selected language row window required')
        if not len(ids):return
        begin=0
        for bank,(source,mapping) in enumerate(zip(self.banks,self.mappings)):
            end=begin+source.shape[1]
            if bank:self._ensure_coarse(bank,source,mapping[ids],begin,end)
            begin=end
        for start in range(0,len(ids),row_window):
            selected=ids[start:start+row_window]
            yield selected,self._selected_shared(selected,prepared_coarse=True)

    def _ensure_coarse(self,bank,source,selected_mapping,begin,end):
        import torch
        if bank not in self.coarse_supplies:
            count=len(source);channels=self.weight.shape[1]
            partial=torch.empty((count,channels),device=source.device,dtype=torch.float16)
            square=torch.empty((count,1),device=source.device,dtype=torch.float32)
            mean=torch.empty_like(square) if self.centered else None
            computed=torch.zeros(count,device=source.device,dtype=torch.bool)
            coefficient=self.weight[begin:end].float().sum(0,keepdim=True) if self.centered else None
            self.coarse_supplies[bank]=(partial,square,mean,coefficient,computed)
        partial,square,mean,coefficient,computed=self.coarse_supplies[bank]
        if self.direct_coarse_membership:
            # Only requested native IDs are consumed by the direct-map
            # merge. Do not construct an unused compact inverse. Every
            # initialization, conflicting max update, scan and compaction
            # is an ordinary captured operation on the real source domain.
            requested=torch.zeros(len(source),device=source.device,dtype=torch.int32)
            requested.scatter_reduce_(0,selected_mapping.long(),
                torch.ones_like(selected_mapping,dtype=torch.int32),reduce='amax',include_self=True)
            used=(requested!=0).nonzero().flatten();inverse=None
        else:
            payload,count=native_index_payload(selected_mapping,len(source))
            used=payload[0,:count];inverse=payload[0,count:]
        needed=used[~computed[used]]
        if len(needed):
            values=source[needed]
            if self.centered:
                values=values.float();average=values.mean(1,keepdim=True)
                mean[needed]=average;values=values-average
            product=values.half()@self.weight[begin:end].half()
            energy=values.float().square().sum(1,keepdim=True)
            partial[needed]=product;square[needed]=energy;computed[needed]=True
        return used,inverse

    def _selected_shared(self,ids,*,prepared_coarse=False,response=None,destination=None):
        """Retain ordinary coarse native products between pilot/output phases."""
        import torch
        import torch.nn.functional as F
        terms=[];begin=0
        for bank,(source,mapping) in enumerate(zip(self.banks,self.mappings)):
            width=source.shape[1];end=begin+width
            if bank==0:
                # Pilot and accepted leaf rows are disjoint; retaining a
                # full finest-domain projection provides no shared work.
                local_weight=torch.cat((self.weight[begin:end],self.weight[-1:]),dim=0)
                terms.extend(_partitioned_terms([source],[mapping[ids]],local_weight,
                    self.centered,compact_native=True))
                begin=end;continue
            selected_mapping=mapping[ids]
            if not prepared_coarse:
                used,inverse=self._ensure_coarse(bank,source,selected_mapping,begin,end)
            partial,square,mean,coefficient,computed=self.coarse_supplies[bank]
            if self.direct_native_maps:
                # The same native IDs used to establish/fill computed rows
                # can address the retained supplies directly. Do not copy
                # wide products/statistics into a second compact domain only
                # for the merge to gather them again through its inverse.
                terms.append((partial,square,mean,coefficient,width,selected_mapping))
            else:
                terms.append((partial[used],square[used],mean[used] if self.centered else None,
                              coefficient,width,inverse))
            begin=end
        merge=_merge_partitioned_terms
        if self.merge_fused:
            from .progressive_triton import merge_affine
            merge=merge_affine
        args=(terms,self.weight,True,self.centered,0,len(ids))
        value=(merge(*args,normalize_output=True,response=response,destination=destination) if response is not None else
               merge(*args,normalize_output=True) if self.fuse_normalization else merge(*args))
        return F.normalize(value,dim=1) if not self.fuse_normalization else value


def partitioned_query(banks,mappings,weight,text,*,centered=False,merge_fused=False,native_preparation=None):
    """Factor a normalized affine query over real native rows.

    Positive source normalization cancels after query normalization. Native
    bank products use FP16 inputs/output and merge in FP32; this changes cast
    order and therefore requires its own captured controls and calibration.
    """
    import torch
    folded=(text.float()@weight.float().T).T
    # F.normalize uses epsilon for a zero source; preserving that source norm
    # also preserves the affine bias after removing the positive common scale.
    return partitioned_affine(banks,mappings,folded,centered=centered,merge_fused=merge_fused,
                              native_preparation=native_preparation)


def equivalent_count(operands,selected):
    """Use a real finest-row ID only if it identifies every operand row.

    No feature similarity, learned equivalence or unbounded associative cache.
    The existing inverse map distributes one row result to its consumers. If
    a hierarchy does not admit this direct key, use per-position execution.
    """
    representatives,_,checks=equivalent_rows(operands,selected)
    return len(representatives),checks


def equivalent_rows(operands,selected):
    """Representative leaf IDs and exact multicast identities, with check work."""
    if not len(selected):return selected,np.empty(0,np.int64),0
    if not operands:return selected,np.arange(len(selected)),0
    finest=max(range(len(operands)),key=lambda i:rows(operands[i][0]))
    ids=np.asarray(operands[finest][1])[selected]
    unique,first,inverse=np.unique(ids,return_index=True,return_inverse=True)
    checks=0
    for _,mapping in operands:
        values=np.asarray(mapping)[selected];checks+=len(values)
        if not np.array_equal(values,values[first][inverse]):return selected,np.arange(len(selected)),checks
    return selected[first],inverse,checks
