# Runtime integration and reference scope


## Reference environment

The original reference used SGLang 0.5.17, an H100 PCIe GPU, and tensor parallelism of 1 (`TP=1`). Its pinned image is:

```text
lmsysorg/sglang@sha256:2fa998888b1b48e778e17f6360a6b60f3cc888b56fa35281669af1d007316337
```

The recorded environment in this image included Torch 2.11.0+cu129 and Transformers 5.12.1. The CPU core does not depend on these packages. This reference does not establish that control contracts have passed on other SGLang versions, with `TP=2`, or with other models.

## Control flow

```text
Quiescent runtime boundary and complete physical observations
  → Construct and validate StrictAllocationEpoch
  → GlobalOptimizer(RecoveryCostModel()).plan_strict(epoch)
  → validate_plan(epoch, plan)
  → StateController.reconcile_strict(epoch, plan, runtime)
  → Bind / Preflight / Native evict_mamba_only / Per-operation postcheck
  → Native Prefix Match / COW / Prefill / Decode
  → Verify cold-run output agreement and state preservation
```

`recurkeep.adapters.sglang.SGLangAdapter` connects to an initialized SGLang cache object and checks native capabilities against the pinned version. It does not start an engine.

`StrictRuntimeBridge` requires the caller to supply three runtime operations:

- `control_epoch`: a real, continuously isolated control window.
- `observe`: complete physical state observations.
- `invalidate`: failure isolation for an invalidated runtime.

An ordinary unlocked context or simulated observations cannot substitute for these operations in a live integration.

All three-state relationships in a `StrictAllocationEpoch` must be present at admission, with no `UNKNOWN` entries. An epoch digest does not prove that the state will remain fresh. Physical binding uses current tokens, nodes, slots, components, references, aliases, and capacity accounting. Mappings from an old engine can support an audit but cannot be used to execute operations in a new engine.

Physical component sharing is resolved before admission. The paper's keep-set union over shared components (Eq (13)) is realized inside the control window as an alias-free contract: every admitted view carries unique slot and component identities, each candidate binds to a distinct node, and `preflight` rejects any Keep/Evict physical overlap before the first eviction.

## Original research hooks

`archives/runtime_reference.tar.gz` preserves original file bytes and historical package names. The following paths refer to contents of that archive:

- `evaluation/serving_control_gpu_gate_20261102/`: validated hooks with compact responses and full validation.
- `evaluation/runtime_control_closure_r2c0/` and `r2c1a/`: observation and control references.
- `evaluation/exact_greedy_paired_gpu_fix_b5d4a/observation.py` (together with the paths above): the paper's non-interfering read-only observation path. It is research integration code in this archive; the core package ships planning, validation, and recurrent-only eviction only, and `SGLangAdapter.inspect_checkpoint` intentionally does not reconstruct logical metadata from a runtime handle.
- `evaluation/strict_compatibility_b1c/`: native-match and copy-on-write (COW) observation references.
- `evaluation/lightweight_runtime_e1/`: existing lightweight control contracts.
- `evaluation/t2g1_three_policy_gpu/adapter.py`, `marconi_evict_t2m/evict_policy.py`, and `kvflow_inspired_retention_cpu/policy.py`: original baseline integrations and selection kernels.
- `flowstate/`: core source at the time of archival. See [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) for its mapping to `recurkeep/`.

The archive contains this project's source files needed by static dependencies and execution locks. It does not contain the external SGLang installation source, model weights, every consumer of historical execution locks, or large runtime logs. It is therefore not a portable, one-command GPU reproduction package.

For a runtime port, establish the target environment's identities and supply the referenced data archives. Run CPU and GPU correctness checks separately. Do not disable the original checks merely to make the old `execute.py` run. See the [experiment guide](experiments.md) for archive preparation and the correctness workflow.

## Preserved checks and limitations

The implementation retains the real full-attention (FA) frontier, complete component and token identities, reference and alias checks, capacity accounting, apply-time freshness checks, original SHA values, per-operation postchecks, native COW, and independent cold-run comparisons.

The compact interface changes only the response projection checked against its consumers; it does not remove content validation. A timing difference between two interface-validation calls is not evidence of a performance improvement.

Cross-workflow tensor lineage remains `PROVENANCE_UNVERIFIED`. Successful native matching does not strengthen that lineage conclusion.
