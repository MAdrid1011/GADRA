"""Finite native-row references for coverage-qualified input intersections.

Intervals replace only verified consecutive row lists. Geometry, version and
view qualification remain the caller's responsibility. Host materialization
is not a feature cache and grants no payload reuse by itself.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RowInterval:
    begin: int
    end: int

    def __len__(self):
        return self.end-self.begin


def prepare(values,ports):
    """Validate all native IDs on existing control lanes, including failures."""
    n=len(values)
    # Read IDs, adjacent subtract/compare, and an AND reduction. No sampled
    # endpoint test can establish that the intervening IDs are consecutive.
    work=4*n+3
    cycles=4*math.ceil(n/ports)+math.ceil(math.log2(max(2,n)))+3
    consecutive=(n>0 and int(values[0])>=0 and int(values[-1])<2**32-1
                 and np.all(values[1:]==values[:-1]+1))
    result=RowInterval(int(values[0]),int(values[-1])+1) if consecutive else values
    # One mode word, followed by either two endpoints or the original IDs.
    words=3 if consecutive else 1+n
    cycles+=math.ceil(words/ports)
    return result,work+words,cycles


def storage_bytes(reference):
    return 12 if isinstance(reference,RowInterval) else 4+4*len(reference)


def intersection(left,right,ports):
    """Return exact sorted shared rows and paid descriptor/intersection work."""
    a=isinstance(left,RowInterval);b=isinstance(right,RowInterval)
    if a and b:
        begin=max(left.begin,right.begin);end=min(left.end,right.end)
        # Two mode/endpoints reads, max/min, compare and bounded dispatch.
        work=10;cycles=2*math.ceil(3/ports)+4
        return np.arange(begin,max(begin,end),dtype=np.int64),work,cycles
    if a or b:
        interval,values=(left,right) if a else (right,left)
        n=len(values)
        work=4*n+3
        cycles=math.ceil(3/ports)+4*math.ceil(n/ports)+2
        return values[(values>=interval.begin)&(values<interval.end)],work,cycles
    depth=math.ceil(math.log2(max(2,len(right)+1)))
    return (np.intersect1d(left,right,assume_unique=True),len(left)*depth+2,
            math.ceil(len(left)/ports)*depth+math.ceil(2/ports))
