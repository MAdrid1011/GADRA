"""Memory byte-enable granules, DMA bursts and vector microcode stages."""
from struct import calcsize

# Result address, dispatch state, join PC, opcode/valid.
DIAGNOSTIC_FORK_DESCRIPTOR_BYTES = calcsize('<QQII')
# Fixed-row checkpoint loop: next row, row limit and current event PC.
DIAGNOSTIC_ROW_LOOP_BYTES = calcsize('<III')
# Output-ID begin/end; two producer completion counters/valid bits.
DIAGNOSTIC_WINDOW_DESCRIPTOR_BYTES = calcsize('<II')
DIAGNOSTIC_DECISION_DESCRIPTOR_BYTES = calcsize('<II')
SECTOR_BYTES = 32
DMA_BURST_BYTES = 256
SLOT_BYTES = 2048
SLOT_SECTORS = SLOT_BYTES // SECTOR_BYTES
BURST_SECTORS = DMA_BURST_BYTES // SECTOR_BYTES
DEPENDENCY_GRANULE_BYTES = SECTOR_BYTES
DRAM_TIME_TICKS = 1024
NORM_PASSES = 8
GELU_PASSES = 10
VECTOR_NORM_PASSES = 3
SOFTMAX_PASSES = 5
