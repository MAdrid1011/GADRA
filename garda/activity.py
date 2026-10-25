"""Observe execution intervals without changing simulated resources.

These host-side measurement records are not accelerator queues or SRAM. Only
aggregate counters are exported; no interval supplies readiness or scheduling.
"""
import numpy as np

from .protocol import DRAM_TIME_TICKS


def sequential_activity(first,second,first_cycles):
    """Compose drained, non-overlapping phases without inventing overlap."""
    if first is None and second is None:return None
    if first is None or second is None:raise ValueError('Both phases must observe the same resources')
    if first.keys()!=second.keys():raise ValueError('Different activity observation schemas')
    result={}
    for name in first:
        a,b=first[name],second[name]
        if name=='scope':result[name]=a
        elif name=='array_vector_cycles':result[name]=(np.asarray(a)+np.asarray(b)).tolist()
        elif name=='last_matrix_completion':
            result[name]=max(a,first_cycles+b if second['observed_matrix_tiles'] else 0)
        elif name=='vector_reservation_overlap_fraction':continue
        elif name=='dram_service_overlap':
            if a['ticks_per_core_cycle']!=b['ticks_per_core_cycle']:
                raise ValueError('Different DDR clock domains')
            result[name]={k:(a[k] if k in ('scope','ticks_per_core_cycle') else
                            (np.asarray(a[k])+np.asarray(b[k])).tolist() if k=='array_vector_dram_ticks' else a[k]+b[k])
                          for k in a}
        else:result[name]=a+b
    result['vector_reservation_overlap_fraction']=result['array_vector_overlap_cycles']/max(1,result['vector_reserved_cycles'])
    return result


class Activity:
    def __init__(self, arrays):
        self.arrays = arrays
        self.matrix = []
        self.vector = []
        self.max_completion = 0
        self.matrix_tiles = 0

    def matrix_buffer(self, count):
        return np.empty((count, 3), dtype=np.int64)

    def add_matrix(self, intervals, completion=None, tiles=None):
        self.matrix_tiles += len(intervals) if tiles is None else tiles
        if len(intervals):
            self.matrix.append(intervals)
            self.max_completion = max(self.max_completion,
                int(intervals[:, 1].max()) if completion is None else int(completion))

    def add_vector(self, begin, end):
        if end > begin:
            self.vector.append((begin, end))

    def summarize(self, cycles, dma=None):
        if self.max_completion > cycles:
            raise AssertionError('Array pipeline or accumulator completion is undrained')
        matrix = np.concatenate(self.matrix) if self.matrix else np.empty((0, 3), np.int64)
        vector = np.asarray(self.vector, np.int64).reshape(-1, 2)
        # Validate individual physical units before deriving overlap. A sum of
        # durations alone would hide an illegal double reservation.
        for unit in range(self.arrays):
            intervals = matrix[matrix[:, 2] == unit, :2]
            self.validate(intervals, cycles)
        self.validate(vector, cycles)
        if len(matrix) and (np.any(matrix[:, 2] < 0) or np.any(matrix[:, 2] >= self.arrays)):
            raise AssertionError('Activity refers to a nonexistent array')
        time = np.r_[0, cycles, matrix[:, 0], matrix[:, 1], vector[:, 0], vector[:, 1]]
        delta = np.zeros((len(time), 2), np.int64)
        start = 2
        delta[start:start + len(matrix), 0] = 1
        start += len(matrix)
        delta[start:start + len(matrix), 0] = -1
        start += len(matrix)
        delta[start:start + len(vector), 1] = 1
        start += len(vector)
        delta[start:start + len(vector), 1] = -1
        order = np.argsort(time, kind='stable')
        times, first = np.unique(time[order], return_index=True)
        active = np.cumsum(np.add.reduceat(delta[order], first, axis=0), axis=0)
        if np.any(active < 0) or np.any(active[:, 0] > self.arrays) or np.any(active[:, 1] > 1):
            raise AssertionError('Activity exceeds physical execution capacity')
        histogram = np.zeros((self.arrays + 1, 2), np.int64)
        np.add.at(histogram, (active[:-1, 0], active[:-1, 1]), np.diff(times))
        array_any = int(histogram[1:].sum())
        vector_reserved = int(histogram[:, 1].sum())
        overlap = int(histogram[1:, 1].sum())
        result = dict(
            scope='Array issue excludes pipeline tails and accumulator holds; vector reservations include microcode waits; no latency decomposition',
            array_vector_cycles=histogram.tolist(),
            array_issue_lane_cycles=int((histogram.sum(axis=1) * np.arange(self.arrays + 1)).sum()),
            array_any_busy_cycles=array_any,
            vector_reserved_cycles=vector_reserved,
            array_vector_overlap_cycles=overlap,
            vector_reservation_overlap_fraction=overlap / max(1, vector_reserved),
            neither_array_nor_vector_cycles=int(histogram[0, 0]),
            last_matrix_completion=self.max_completion,
            observed_matrix_tiles=self.matrix_tiles,
            observed_array_issue_intervals=len(matrix), observed_vector_programs=len(vector))
        if dma is not None:
            result['dram_service_overlap'] = self.dma_overlap(matrix, vector, dma, cycles, histogram)
        return result

    def dma_overlap(self, matrix, vector, dma, cycles, compute_histogram):
        """Exact joint service intervals in the existing fractional DDR domain."""
        ticks = DRAM_TIME_TICKS
        dma = np.asarray(dma, np.int64).reshape(-1, 2)
        self.validate(dma, cycles * ticks)
        time = np.r_[0, cycles * ticks, matrix[:, 0] * ticks, matrix[:, 1] * ticks,
                     vector[:, 0] * ticks, vector[:, 1] * ticks, dma[:, 0], dma[:, 1]]
        delta = np.zeros((len(time), 3), np.int64)
        start = 2
        for component, count in ((0, len(matrix)), (1, len(vector)), (2, len(dma))):
            delta[start:start + count, component] = 1
            start += count
            delta[start:start + count, component] = -1
            start += count
        order = np.argsort(time, kind='stable')
        times, first = np.unique(time[order], return_index=True)
        active = np.cumsum(np.add.reduceat(delta[order], first, axis=0), axis=0)
        if (np.any(active < 0) or np.any(active[:, 0] > self.arrays)
                or np.any(active[:, 1:] > 1) or np.any(active[-1])):
            raise AssertionError('Joint activity exceeds physical resources or is undrained')
        histogram = np.zeros((self.arrays + 1, 2, 2), np.int64)
        np.add.at(histogram, tuple(active[:-1, i] for i in range(3)), np.diff(times))
        if not np.array_equal(histogram.sum(axis=2), compute_histogram * ticks):
            raise AssertionError('DDR observation changed compute intervals')
        service = int(histogram[:, :, 1].sum())
        array = int(histogram[1:, :, 1].sum())
        vector = int(histogram[:, 1, 1].sum())
        both = int(histogram[1:, 1, 1].sum())
        return dict(scope='DDR bus service only, excluding request/return latency; not an additive latency decomposition',
            ticks_per_core_cycle=ticks, array_vector_dram_ticks=histogram.tolist(),
            dram_service_ticks=service, dram_with_array_issue_ticks=array,
            dram_with_vector_reservation_ticks=vector, dram_with_both_compute_ticks=both,
            dram_with_either_compute_ticks=array + vector - both,
            dram_without_array_or_vector_ticks=int(histogram[0, 0, 1]),
            no_array_vector_or_dram_service_ticks=int(histogram[0, 0, 0]),
            recorded_bus_runs=len(dma))

    @staticmethod
    def validate(intervals, cycles):
        if not len(intervals):
            return
        intervals = intervals[np.argsort(intervals[:, 0], kind='stable')]
        if (np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] < intervals[:, 0])
                or np.any(intervals[:, 1] > cycles)
                or np.any(intervals[1:, 0] < intervals[:-1, 1])):
            raise AssertionError('Execution intervals overlap or exceed completion')
