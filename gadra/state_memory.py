"""Bounded history and response storage sharing the modeled DRAM port."""
from collections import OrderedDict
import math
import numpy as np
from .protocol import DMA_BURST_BYTES, SECTOR_BYTES


class StateStore:
    def __init__(self, memory, entries, page_bytes, name):
        self.memory=memory;self.entries=entries;self.page_bytes=page_bytes;self.name=name
        self.pages=OrderedDict()

    def access(self, sectors, *, write=False, ready=0):
        m=self.memory;done=max(ready,m.clock);per_page=self.page_bytes//SECTOR_BYTES
        for page in np.unique(np.asarray(sectors,dtype=np.int64)//per_page):
            selected=np.asarray(sectors)[np.asarray(sectors)//per_page==page]%per_page
            mask=sum(1<<int(i) for i in np.unique(selected))
            if int(page) in self.pages:
                valid,dirty,available=self.pages.pop(int(page));m.stats[self.name+'_hits']+=1
            else:
                valid=dirty=0;available=0;m.stats[self.name+'_misses']+=1
                if len(self.pages)==self.entries:
                    _,(old_valid,old_dirty,old_ready)=self.pages.popitem(last=False)
                    done=max(done,self.flush_page(old_dirty,max(done,old_ready)))
                    m.stats[self.name+'_evictions']+=1
            if write:
                valid|=mask;dirty|=mask;available=max(available,ready)
            else:
                for offset in range(0,per_page,DMA_BURST_BYTES//SECTOR_BYTES):
                    bits=((1<<(DMA_BURST_BYTES//SECTOR_BYTES))-1)<<offset
                    if mask&bits&~valid:
                        available=max(available,self.transfer(False,max(done,m.clock)))
                        valid|=bits
            done=max(done,available)+math.ceil(len(selected)/m.c['cam_lookups_per_cycle'])
            self.pages[int(page)]=(valid,dirty,done)
        m.clock=max(m.clock,done)
        m.stats[self.name+'_peak_bytes']=max(m.stats[self.name+'_peak_bytes'],len(self.pages)*self.page_bytes)
        m.stats['control_sram_write_bytes' if write else 'control_sram_read_bytes']+=len(sectors)*SECTOR_BYTES
        return done

    def transfer(self, write, ready):
        m=self.memory
        done=m.core.transaction(ready,write,m.state,m.pending,m.counts,m.burst_cycles,
            m.c['dram_latency_cycles'],DMA_BURST_BYTES,m.dma_activity)
        m.stats[self.name+('_dram_write_bytes' if write else '_dram_read_bytes')]+=DMA_BURST_BYTES
        return int(done)

    def flush_page(self, dirty, ready):
        for offset in range(0,self.page_bytes//SECTOR_BYTES,DMA_BURST_BYTES//SECTOR_BYTES):
            if dirty&(((1<<(DMA_BURST_BYTES//SECTOR_BYTES))-1)<<offset):
                ready=max(ready,self.transfer(True,ready))
        return ready

    def discard(self,sectors):
        per_page=self.page_bytes//SECTOR_BYTES
        for sector in np.unique(sectors):
            page=int(sector)//per_page
            if page in self.pages:
                valid,dirty,ready=self.pages[page];bit=1<<(int(sector)%per_page)
                valid&=~bit;dirty&=~bit
                if valid:self.pages[page]=(valid,dirty,ready)
                else:del self.pages[page]

    def finish(self, ready):
        for valid,dirty,available in self.pages.values():
            ready=max(ready,self.flush_page(dirty,max(ready,available)))
        self.pages.clear()
        return ready
