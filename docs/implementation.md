# Implementation comparison

## Resources

The default profile serializes scalar microprograms with `vector_lanes = 1`.
Its independent scalar path has one FP MAC and one iterative FP unit. Matrix
work uses the separate 64 × 64 array. Ordinary scalar operations, reductions,
diagnostics, native index construction, coverage ordering, and dependency
updates all enter the same scalar reservation ledger. Compound programs retain
ownership across their memory waits. The ledger gives each reservation an
operator, domain, begin cycle, end cycle, and resource identity.

`statistics.vector_cycles` counts arithmetic microprogram work; a reserved
interval can be longer because of memory waits. These are separate counters.
Iterative instruction timing and pipeline initiation intervals belong to the
implementation mapping. A source-level architectural replay alone does not
establish physical throughput or PPA.

## Private resource contract

Each paper-evaluation fragment names an `implementation_contract` JSON file
in its local study manifest. It uses schema `gadra-implementation/v1` and has:

- `rtl_revision` and `physical_revision`: implementation identities.
- `config_sha256`: `gadra.profile.fingerprint()` of the complete replay config.
- `resources`: `vector_lanes = 1`, `scalar_fp_mac_units = 1`,
  `scalar_iterative_units = 1`, and `shares_matrix_array = false`.
- `source_files`: hardware source `path` and `sha256` pairs. Paths resolve
  relative to the contract. RTL source hashes are checked before aggregation.
- `operator_mapping`: one entry for every recorded scalar operator. Each entry
  names its `rtl_module` and the `units` it uses, drawn from `scalar_fp_mac`,
  `scalar_iterative_fp`, and `scalar_control`.

The mapping binds native ATen operations and internal programs such as
`diagnostic_unique_metadata`, `inverse_row_metadata`, `integer_counter_updates`,
`coverage_preparation`, `coverage_lift_preparation`, and
`dependency_reference_update` to their actual sequencer programs. The same
mapping identifies iterative instructions and their RTL latency. Hardware
sources, implementation records, and physical reports remain private inputs.

## Paired execution record

`rtl_trace` names a private JSON file exported by the implementation harness.
Its schema is `gadra-rtl-trace/v1`; `producer` is `verilator`. It supplies the
same `rtl_revision`, `config_sha256`, and `capture_sha256` as the replay.
`gadra.implementation.capture_digest()` hashes the native graph, dependency
maps, captured observations, and geometry to identify the fragment.

The file has a `variants` object for the requested variants. Each variant
contains `cycles`, `vector_work_cycles`, and `audit_events`. Events use these
fields:

- `native_issue`: native `event`, admitted `positions`, and admission `cycle`.
- `scalar_issue`: `operator`, `domain`, `begin`, `end`, and
  `resource = independent_scalar_sequencer`.
- `checkpoint_commit`: checkpoint `depth`, exit `positions`, and commit `cycle`.
- `task_cancel`: native `event`, canceled `positions`, cancellation `cycle`,
  triggering `depth`, and `decision_cycle`.
- `native_outputs_commit`: final `depth`, remaining `positions`, and `cycle`.

`validate_pair()` compares total cycles, scalar work cycles, scalar reservations,
admitted task instances, committed exits, final outputs, and cancellation sets.
It rejects scalar overlap, cancellation before a decision, cancellation of
issued work, and later admission of a canceled task. The validation status
changes only after every requested variant matches. Generate this record from
the implementation harness; copying replay output is not an RTL comparison.

For architectural exploration, add `--model-only` to `gadra evaluate`. It uses
the same causal execution path and emits `architectural_model_only` status.
The default paper aggregation requires the paired implementation inputs.
All generated ledgers and comparisons stay in ignored local directories.
