"""Finite non-preemptive reservations for the existing vector issue queue."""
from bisect import insort
from struct import Struct


class ReadyQueue:
    # Ready/end clocks, task-owner pointer and opcode/valid bits. Deployed
    # entries use the declared control SRAM; the Python tuples are not RAM.
    entry_bytes=Struct('<QQII').size
    def __init__(self,capacity):
        if capacity<=0:raise ValueError('Ready queue must have entries')
        self.capacity=capacity
        self.pending=[]
        self.floor=0
        self.horizon=0
        self.peak=0
        self.backpressure=0
        self.hidden=0

    def _admission(self,issued):
        floor=max(issued,self.floor)
        pending=[p for p in self.pending if p[1]>floor]
        if len(pending)>=self.capacity:
            floor=min(end for _,end in pending)
            pending=[p for p in pending if p[1]>floor]
        return floor,pending

    @staticmethod
    def _start(earliest,duration,floor,pending):
        begin=max(earliest,floor)
        for start,end in pending:
            if begin+duration<=start:break
            if begin<end:begin=end
        return begin

    def preview(self,earliest,duration,issued,admission_cycles=0):
        """Non-mutating earliest start for bounded multi-unit arbitration."""
        if not duration:return max(earliest,issued,self.floor)+admission_cycles
        floor,pending=self._admission(issued)
        return self._start(earliest,duration,floor+admission_cycles,pending)

    def reserve(self,earliest,duration,issued,admission_cycles=0):
        """Book one contiguous program; retain future owners until completion.

        Admission is in issue order. Execution may use a hole before an older
        program whose operands are not ready. On a full queue, admission waits
        for its earliest completion; no reservation is silently dropped.
        """
        if duration<0 or admission_cycles<0:raise ValueError('Negative resource duration')
        previous_floor=max(issued,self.floor)
        if not duration:
            self.floor=previous_floor+admission_cycles
            self.pending=[p for p in self.pending if p[1]>self.floor]
            return max(earliest,self.floor)
        floor,self.pending=self._admission(issued)
        self.backpressure+=floor-previous_floor
        self.floor=floor+admission_cycles
        begin=self._start(earliest,duration,self.floor,self.pending)
        finish=begin+duration
        if duration:
            self.hidden+=max(0,min(finish,self.horizon)-begin)
            insort(self.pending,(begin,finish))
            self.horizon=max(self.horizon,finish)
            self.peak=max(self.peak,len(self.pending))
        return finish
