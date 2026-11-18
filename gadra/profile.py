"""Executable constants for the manuscript's 1 GHz, 152 KiB design."""
import hashlib
import json
from .protocol import DMA_BURST_BYTES, SLOT_BYTES

PROFILE = 'paper-v1'
HARDWARE_FIELDS = ('clock_hz', 'arrays', 'array_rows', 'array_cols', 'fp32_mac_cycles',
    'feature_sram_bytes', 'weight_sram_bytes', 'accumulator_register_bytes',
    'output_sram_bytes', 'control_sram_bytes', 'history_sram_bytes',
    'control_metadata_bytes', 'cam_entries', 'resident_slot_bytes', 'dma_burst_bytes',
    'dram_bandwidth_bytes_per_second', 'dram_latency_cycles')


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(config, *, sensitivity=False):
    expected = dict(profile=PROFILE, clock_hz=1_000_000_000, arrays=1, array_rows=64,
        array_cols=64, fp32_mac_cycles=1, weight_sram_bytes=32*1024,
        accumulator_register_bytes=16*1024, output_sram_bytes=16*1024,
        control_sram_bytes=80*1024, history_sram_bytes=64*1024,
        control_metadata_bytes=16*1024, resident_slot_bytes=128, dma_burst_bytes=64,
        dram_bandwidth_bytes_per_second=204_800_000_000, dram_latency_cycles=180,
        predictor_window=2, query_normalization='l1', correction_horizon='cumulative_l1',
        predictor_bins=4, predictor_quantile=.95, predictor_min_samples=64,
        history_active_positions=64, max_queries=256, response_directory_entries=64)
    if not sensitivity:
        expected.update(cam_entries=64, feature_sram_bytes=8*1024, semantic_tolerance=.02)
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f'Manuscript profile requires {name}={value}, got {config.get(name)}')
    if config['feature_sram_bytes'] != config['cam_entries']*SLOT_BYTES:
        raise ValueError('Each CAM entry must own one 128 B feature slot')
    if config['resident_slot_bytes'] != SLOT_BYTES or config['dma_burst_bytes'] != DMA_BURST_BYTES:
        raise ValueError('Configured granules must equal the executed memory protocol')
    storage = sum(config[k] for k in ('feature_sram_bytes','weight_sram_bytes',
        'accumulator_register_bytes','output_sram_bytes','control_sram_bytes'))
    if not sensitivity and storage != 152*1024:
        raise ValueError('The manuscript storage budget is 152 KiB')
    return dict(profile=PROFILE, config_sha256=fingerprint(config), total_storage_bytes=storage,
        peak_fp32_flops=8_192_000_000_000,
        hardware={k:config[k] for k in HARDWARE_FIELDS})
