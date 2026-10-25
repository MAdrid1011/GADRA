"""Compiler row provenance for B; identities come from real views and maps.

The ndarrays are a host representation of existing leaf/parent maps, not an
additional on-chip row table. Only proven two-dimensional domains propagate.
Equal row counts, opaque reshapes and many-to-one inverse guesses do not.
"""
import math

import numpy as np

from .graph import key,rows
from .protocol import DMA_BURST_BYTES


# Off-chip per-view binding: a 32-bit program offset and native row count.
LAYOUT_BINDING_BYTES = 8
LAYOUT_BINDING_OPS = 2


class CoverageRows:
    def __init__(self,graph,leaf_order,packing,links=None,link_limit=None,*,attention_links=False,
                 layout_domain_reuse=False,link_domain_reuse=False):
        self.g=graph;self.order=leaf_order;self.packing=packing
        self.links=links;self.link_limit=link_limit;self.native_links={}
        self.attention_outputs={(key(s),self.signature(s)) for e in graph.events if e['kind']=='sdpa'
                                for s in e['outputs']} if attention_links else set()
        # Per-query lowering: selected ID reads, CSR ID reads, converted ID
        # writes (32-bit words), and elementary vector operations. These are
        # consumed and paid by the caller; not a persistent hardware table.
        self.query_work=np.zeros(4,np.int64)
        if links is not None:
            offsets,neighbors=links
            if (not np.issubdtype(offsets.dtype,np.integer) or
                not np.issubdtype(neighbors.dtype,np.integer) or
                offsets.shape!=(len(leaf_order)+1,) or neighbors.ndim!=1 or
                offsets[0]!=0 or offsets[-1]!=len(neighbors) or np.any(np.diff(offsets)<0) or
                np.any(neighbors<0) or np.any(neighbors>=len(leaf_order)) or
                not isinstance(link_limit,int) or isinstance(link_limit,bool) or link_limit<1 or
                np.any(np.diff(offsets)>link_limit)):
                raise ValueError('Coverage links require bounded CSR in the real leaf domain')
        self.views={};self.orders={};self.scalar_ops=0;self.metadata_bytes=0
        self.layout_domain_reuse=layout_domain_reuse
        # Compiler identities, not an on-chip associative cache. A retained
        # mapping reference prevents Python object-ID reuse. Only existing
        # provenance propagation can pass the same mapping object onward.
        # Feature versions, native links and data residency remain separate.
        self.layout_domains={};self.layout_domain_reuses=0
        # An immutable geometry program can serve proven row-local feature
        # versions. This shares off-chip adjacency construction, never the
        # versioned feature references checked by linked_frontier().
        self.link_domain_reuse=link_domain_reuse
        self.native_link_domains={};self.native_link_domain_reuses=0
        for cp in graph.maps.get('coverage_checkpoints',graph.maps.get('checkpoints',[])):
            if 'spec' in cp:self.add(cp['spec'],cp['mapping'])
            for spec,mapping in zip(cp.get('readout_inputs',[]),cp.get('readout_maps',[])):
                self.add(spec,mapping)
        while True:
            count=sum(map(len,self.views.values()))
            self.reverse()
            self.forward()
            if sum(map(len,self.views.values()))==count:break

    @staticmethod
    def row_local(event):
        # A mapped vector scatter executes in compact consumer ordinals but
        # writes global destination rows. Its vector opcode does not prove a
        # row-local input/output identity across that partial update.
        if event['kind']=='indexed_store':return False
        if event.get('indexed_source',False):return False
        return event['kind']=='linear' or event.get('op','').startswith((
            'aten.add.','aten.add_.','aten.sub.','aten.sub_.','aten.mul.','aten.mul_.',
            'aten.div.','aten.div_.','aten.relu','aten.batch_norm',
            'aten.gelu','aten.layer_norm','aten.to.','aten.clamp_min',
            'aten.linalg_vector_norm','aten.clone','aten.contiguous'))

    @staticmethod
    def private_rows(event):
        """Prove no feature byte is shared by distinct output rows.

        These tasks still use B's weight batches. Coverage scoring has no
        cross-row feature to retain, so it must not destroy native locality.
        Unknown views, broadcasts, gathers and neighborhood maps stay general.
        """
        if not CoverageRows.row_local(event) or len(event['outputs'])!=1:return False
        out=event['outputs'][0]
        if len(out['shape'])!=2:return False
        if event['kind']=='linear' and any(s['role']=='feature' for s in event['inputs'][1:]):return False
        if any(name in event for name in ('diagnostic_control_rows','diagnostic_source_rows')):return False
        if event.get('op','').startswith('aten.layer_norm'):
            # ATen requires affine weight.shape == normalized_shape. A real
            # one-dimensional channel weight therefore proves row-local
            # normalization; absent affine parameters cannot prove its axes.
            if (len(event['inputs'])<2 or
                any(s['role']!='weight' or s['shape']!=[out['shape'][1]]
                    or s['stride']!=[1] for s in event['inputs'][1:])):return False
        sources=event['inputs'][:1] if event['kind']=='linear' else event['inputs']
        features=[s for s in sources if s['role']=='feature']
        if not features:return False
        views={}
        for spec in features:
            if (len(spec['shape'])!=2 or rows(spec)!=rows(out) or
                spec['stride'][1]!=1 or spec['stride'][0]<spec['shape'][1]):return False
            identity=(spec['address'],tuple(spec['shape']),tuple(spec['stride']),spec['element_bytes'])
            if key(spec) in views and views[key(spec)]!=identity:return False
            views[key(spec)]=identity
        return True

    def compatible(self,spec,mapping):
        known=self.lookup(spec)
        return known is None or np.array_equal(known,mapping)

    def reverse(self):
        """Backward row identity only through proven row-local or native maps.

        This composes geometric domains, never copies feature values or
        infers identity merely from equal tensor lengths.
        """
        for index in range(len(self.g.events)-1,-1,-1):
            e=self.g.events[index]
            if len(e['outputs'])!=1:continue
            out=e['outputs'][0]
            if len(out['shape'])!=2:continue
            known=self.lookup(out);inputs=e['inputs'];op=e.get('op','')
            if known is not None and self.row_local(e):
                sources=inputs[:1] if e['kind']=='linear' else inputs
                candidates=[s for s in sources if s['role']=='feature'
                            and len(s['shape'])==2 and rows(s)==rows(out)]
                if all(self.compatible(s,known) for s in candidates):
                    for s in candidates:
                        if self.lookup(s) is None:self.add(s,known)
            elif known is not None and inputs and (e.get('indexed_source',False) or
                    (op=='aten.index.Tensor' and e.get('row_selector'))):
                source=inputs[0];maps=self.g.maps['indices'].get(index,[])
                if len(source['shape'])!=2 or len(maps)!=1:continue
                selected=np.asarray(maps[0]).reshape(-1)
                if len(selected)!=rows(out):continue
                selected=np.where(selected<0,selected+rows(source),selected)
                if np.any(selected<0) or np.any(selected>=rows(source)):continue
                mapping=np.full(len(known),-1,np.int64);valid=known>=0
                mapping[valid]=selected[known[valid]]
                if self.compatible(source,mapping) and self.lookup(source) is None:self.add(source,mapping)
            elif e['kind']=='spconv' and inputs:
                source=inputs[0];rule=self.g.maps['maps'][e['map_key']]
                if not rule.get('native_submanifold_expanded',False):continue
                matrix=np.asarray(rule['mapping'])
                if (len(source['shape'])!=2 or rows(source)!=rows(out) or len(matrix)%2!=1
                    or not np.array_equal(matrix[len(matrix)//2],np.arange(rows(out)))):continue
                prior=self.lookup(source)
                if known is not None and prior is None:self.add(source,known)
                elif prior is not None and known is None:self.add(out,prior)

    def forward(self):
        for index,event in enumerate(self.g.events):
            if len(event['outputs'])!=1:continue
            out=event['outputs'][0]
            if len(out['shape'])!=2 or self.lookup(out) is not None:continue
            sources=event['inputs'][:1] if event['kind']=='linear' else event['inputs']
            inputs=[s for s in sources if s['role']=='feature']
            op=event.get('op','')
            local=self.row_local(event)
            candidates=[]
            if local:
                candidates=[m for s in inputs if rows(s)==rows(out)
                            for m in [self.lookup(s)] if m is not None]
            elif event.get('indexed_source',False) or (op=='aten.index.Tensor' and event.get('row_selector')):
                mapping=self.g.maps['indices'].get(index,[])
                if inputs and len(mapping)==1:
                    found=self.inverse(inputs[0],np.asarray(mapping[0]).reshape(-1))
                    if found is not None:candidates=[found]
            elif event['kind']=='ancestor_merge':
                # The fine native bank has an injective real leaf map. A
                # coarse bank with repeated consumers cannot establish which
                # descendant output row is meant and is never inverted.
                mapping=self.g.maps['indices'][index][0]
                found=self.inverse(event['inputs'][0],np.asarray(mapping))
                if found is not None:candidates=[found]
            elif op=='aten.cat.default' and inputs:
                mappings=[self.lookup(s) for s in inputs]
                if (all(len(s['shape'])==2 and rows(s)==rows(out) for s in inputs)
                    and sum(s['shape'][1] for s in inputs)==out['shape'][1]):
                    candidates=[m for m in mappings if m is not None]
                elif all(m is not None for m in mappings):
                    if sum(rows(s) for s in inputs)==rows(out):
                        combined=np.full(len(self.order),-1,np.int64);begin=0;valid=True
                        for source,mapping in zip(inputs,mappings):
                            active=mapping>=0
                            if np.any((combined>=0)&active):valid=False;break
                            combined[active]=mapping[active]+begin;begin+=rows(source)
                        if valid:candidates=[combined]
            elif op=='aten.copy_.default' and len(inputs)==2:
                mapping=self.lookup(inputs[1])
                if mapping is not None and rows(inputs[1])==rows(out):
                    candidates=[mapping]
                    self.copy_domain(event,mapping)
            if candidates and all(np.array_equal(candidates[0],m) for m in candidates[1:]):
                self.add(out,candidates[0])


    def copy_domain(self,event,mapping):
        """Keep row identity across real partial writes, never their values."""
        out=event['outputs'][0];storage=self.g.storage[out['storage']]
        pitch=out['stride'][0]*out['element_bytes']
        if (pitch<=0 or out['stride'][1]!=1 or storage['bytes']%pitch):return
        offset=out['address']-storage['address'];begin,column=divmod(offset,pitch)
        if column+out['shape'][1]*out['element_bytes']>pitch:return
        whole=dict(out,address=storage['address'],shape=[storage['bytes']//pitch,pitch//out['element_bytes']],
                   stride=[pitch//out['element_bytes'],1])
        if begin+rows(out)>rows(whole):return
        old=dict(whole,version=event['inputs'][0]['version'])
        previous=self.lookup(old)
        combined=np.full(len(mapping),-1,np.int64) if previous is None else previous.copy()
        valid=mapping>=0;updated=mapping+begin
        touched=(combined>=begin)&(combined<begin+rows(out))
        if (np.any(valid&(combined>=0)&(combined!=updated)) or
            np.any(touched&((~valid)|(combined!=updated)))):
            # A copy between two different point orderings is legal, but a
            # single canonical point domain for that storage is not proven.
            return
        combined[valid]=updated[valid]
        self.add(whole,combined)

    def add(self,spec,mapping):
        if len(spec['shape'])!=2 or spec['role']!='feature':return
        mapping=np.asarray(mapping)
        if mapping.shape!=(len(self.order),) or np.any(mapping<-1) or np.any(mapping>=rows(spec)):
            raise ValueError('Coverage provenance is outside its captured native row domain')
        records=self.views.setdefault(key(spec),[])
        signature=self.signature(spec)
        for view,other in records:
            if self.signature(view)==signature:
                if not np.array_equal(mapping,other):raise ValueError('Conflicting native row provenance')
                return
        records.append((spec,mapping))

    @staticmethod
    def signature(spec):
        return spec['address'],tuple(spec['shape']),tuple(spec['stride']),spec['element_bytes']

    def lookup(self,spec):
        if len(spec['shape'])!=2:return None
        for view,mapping in self.views.get(key(spec),[]):
            if self.signature(view)==self.signature(spec):return mapping
            if view['element_bytes']!=spec['element_bytes']:continue
            pitch=view['stride'][0]*view['element_bytes']
            if pitch<=0 or spec['stride'][0]*spec['element_bytes']!=pitch:continue
            delta=spec['address']-view['address']
            if delta<0:continue
            begin,column=divmod(delta,pitch)
            span=((spec['shape'][1]-1)*spec['stride'][1]+1)*spec['element_bytes']
            view_span=((view['shape'][1]-1)*view['stride'][1]+1)*view['element_bytes']
            if (view['stride'][1]<=0 or column%(view['stride'][1]*view['element_bytes'])
                or spec['stride'][1]!=view['stride'][1] or column+span>view_span
                or begin+rows(spec)>rows(view)):continue
            return np.where((mapping>=begin)&(mapping<begin+rows(spec)),mapping-begin,-1)
        return None

    def inverse(self,source,selected):
        mapping=self.lookup(source)
        if mapping is None:return None
        selected=np.where(selected<0,selected+rows(source),selected)
        if (np.any(selected<0) or np.any(selected>=rows(source)) or
            len(np.unique(selected))!=len(selected)):return None
        inverse=np.full(rows(source),-1,np.int64);inverse[selected]=np.arange(len(selected))
        result=np.full(len(mapping),-1,np.int64);valid=mapping>=0
        result[valid]=inverse[mapping[valid]]
        return result

    def layout(self,spec):
        signature=(key(spec),self.signature(spec))
        if signature in self.orders:return self.orders[signature]
        mapping=self.lookup(spec)
        if mapping is None:return None
        domain=(id(mapping),rows(spec))
        if self.layout_domain_reuse and domain in self.layout_domains:
            proof,candidate=self.layout_domains[domain]
            if proof is not mapping:raise AssertionError('Reused compiler mapping identity')
            self.orders[signature]=candidate
            # Each new feature view still reads its complete off-chip order
            # and its own binding; only duplicate construction is removed.
            self.metadata_bytes+=candidate.nbytes+LAYOUT_BINDING_BYTES
            self.scalar_ops+=LAYOUT_BINDING_OPS
            self.layout_domain_reuses+=1
            return candidate
        ordered=mapping[self.order];ordered=ordered[ordered>=0]
        _,first=np.unique(ordered,return_index=True)
        candidate=ordered[np.sort(first)]
        if len(candidate)!=rows(spec):return None
        chunks=candidate//self.packing
        _,first=np.unique(chunks,return_index=True)
        candidate=(chunks[np.sort(first),None]*self.packing+np.arange(self.packing)).ravel()
        candidate=candidate[candidate<rows(spec)]
        if not np.array_equal(np.sort(candidate),np.arange(rows(spec))):
            raise ValueError('Proven coverage order omitted a native output row')
        self.orders[signature]=candidate
        # Build/read lifted scheduling metadata, separately from the existing
        # geometry cost. Sorting is conservatively a compare/move program;
        # native mapping values do not constitute a free hardware permutation.
        self.scalar_ops+=len(mapping)+2*len(ordered)*math.ceil(math.log2(max(2,len(ordered))))
        self.metadata_bytes+=mapping.nbytes+candidate.nbytes
        if self.layout_domain_reuse:
            candidate.setflags(write=False)
            self.layout_domains[domain]=(mapping,candidate)
            self.metadata_bytes+=LAYOUT_BINDING_BYTES
            self.scalar_ops+=LAYOUT_BINDING_OPS
        return candidate

    def neighbors(self,spec,selected,*,deduplicate=True):
        """Read bounded native adjacency, backed by an off-chip CSR program.

        Endpoints are lifted only by proven leaf/native identities. Links
        absorbed into one coarse position are removed. Native degree is
        capped again; a coarse position does not gain an unbounded queue.
        Returns IDs, logical bytes and two dependent burst-address lists.
        The offset and neighbor buffers each begin at a burst boundary.
        """
        empty=np.empty(0,np.int64)
        if self.links is None:return empty,0,(empty,empty)
        if (key(spec),self.signature(spec)) in self.attention_outputs:
            # Captured SDPA writes [patch,head,query,channel], interleaved by
            # head within each query. Its exact flat alias is propagated from
            # the real downstream depadding map, not from equal row counts.
            if len(spec['shape'])!=4 or not len(selected):return empty,0,(empty,empty)
            patches,heads,sequence,dim=spec['shape']
            if tuple(spec['stride'])!=(sequence*heads*dim,dim,heads*dim,1):return empty,0,(empty,empty)
            group=np.unique(selected//sequence)
            if len(group)!=1:return empty,0,(empty,empty)
            flat=dict(spec,shape=[patches*sequence,heads*dim],stride=[heads*dim,1])
            if self.lookup(flat) is None:return empty,0,(empty,empty)
            tokens=selected//(heads*sequence)*sequence+selected%sequence
            payload,nbytes,bursts=self.neighbors(flat,tokens,deduplicate=deduplicate)
            # Links stay within the querying head identity. Cross-patch
            # candidates still require actual same-version input sharing.
            payload=payload.astype(np.int64)
            converted=(payload//sequence*heads+int(group[0]%heads))*sequence+payload%sequence
            self.query_work+=np.array([2*len(selected),len(payload),2*len(payload),
                                      8*len(selected)+12*len(payload)],np.int64)
            return converted,nbytes,bursts
        signature=(key(spec),self.signature(spec))
        if signature not in self.native_links:
            mapping=self.lookup(spec)
            if mapping is None:return empty,0,(empty,empty)
            domain=(id(mapping),rows(spec))
            if self.link_domain_reuse and domain in self.native_link_domains:
                proof,adjacency=self.native_link_domains[domain]
                if proof is not mapping:raise AssertionError('Reused compiler mapping identity')
                self.native_links[signature]=adjacency
                self.metadata_bytes+=LAYOUT_BINDING_BYTES
                self.scalar_ops+=LAYOUT_BINDING_OPS
                self.native_link_domain_reuses+=1
                # Query-time offset and neighbor DMA below remains per view.
                # A geometry binding cannot qualify actual feature sharing.
            else:self.build_native_links(signature,spec,mapping,domain)
        offsets,neighbors=self.native_links[signature]
        positions=[np.arange(offsets[row],offsets[row+1],dtype=np.int64) for row in selected]
        positions=np.concatenate(positions) if positions else empty
        payload=neighbors[positions]
        offset_words=np.unique(np.r_[selected,selected+1])
        offset_bursts=np.unique(offset_words*8//DMA_BURST_BYTES)
        neighbor_bursts=np.unique(positions*4//DMA_BURST_BYTES)
        return (np.unique(payload) if deduplicate else payload),offset_words.nbytes+payload.nbytes,(offset_bursts,neighbor_bursts)

    def build_native_links(self,signature,spec,mapping,domain):
        """Build one bounded off-chip geometry program, with explicit fees."""
        offsets,neighbors=self.links
        left=np.repeat(mapping,np.diff(offsets));right=mapping[neighbors]
        valid=(left>=0)&(right>=0)&(left!=right)
        pairs=np.unique(np.stack((left[valid],right[valid]),axis=1),axis=0)
        if len(pairs):
            starts=np.r_[0,np.flatnonzero(np.diff(pairs[:,0]))+1]
            rank=np.arange(len(pairs))-np.repeat(starts,np.diff(np.r_[starts,len(pairs)]))
            pairs=pairs[rank<self.link_limit]
        degree=np.bincount(pairs[:,0],minlength=rows(spec)) if len(pairs) else np.zeros(rows(spec),np.int64)
        native_offsets=np.r_[0,np.cumsum(degree)].astype(np.int64)
        native_neighbors=pairs[:,1].astype(np.int32)
        adjacency=(native_offsets,native_neighbors)
        self.native_links[signature]=adjacency
        # Count gather, filter, sorted dedup/cap and CSR construction.
        self.scalar_ops+=len(mapping)+len(neighbors)*4+len(pairs)*2
        self.scalar_ops+=2*int(valid.sum())*math.ceil(math.log2(max(2,int(valid.sum()))))
        self.metadata_bytes+=mapping.nbytes+offsets.nbytes+neighbors.nbytes
        self.metadata_bytes+=native_offsets.nbytes+native_neighbors.nbytes
        if self.link_domain_reuse:
            native_offsets.setflags(write=False);native_neighbors.setflags(write=False)
            self.native_link_domains[domain]=(mapping,adjacency)
            self.metadata_bytes+=LAYOUT_BINDING_BYTES
            self.scalar_ops+=LAYOUT_BINDING_OPS
