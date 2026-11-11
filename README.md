# RecurKeep

**Efficient Recurrent Checkpoint Management for Agentic Hybrid LLM Serving**

RecurKeep is a recurrent-checkpoint management framework for long-lived agentic workflows served by hybrid large language models (LLMs). It allocates a shared budget of recurrent checkpoints across pending workflow continuations according to their **executable recovery benefits**. The repository contains the allocator, validation and runtime-control components, experiment inputs, and result records used to reproduce and examine the evaluation.

## Motivation and approach

Hybrid LLMs maintain attention-side cache state alongside recurrent or linear-attention state. An attention-side prefix-cache hit alone does not guarantee executable recovery: resuming a continuation also requires a complete recurrent checkpoint at a compatible position in its execution history.

In branching workflows, a checkpoint on shared history may benefit several continuations, while multiple checkpoints for the same continuation may cover overlapping recovery work. RecurKeep therefore evaluates checkpoint retention **jointly**, rather than ranking checkpoints only by their standalone value.

RecurKeep follows four steps:

1. **Construct an executable-state view.** Combine workflow histories and read-only runtime observations to identify legal checkpoint–continuation relations and executable recovery frontiers.
2. **Estimate recovery benefits.** Use a position-aware recovery-cost model to evaluate the computation avoided by each compatible checkpoint.
3. **Allocate under a shared budget.** Maximize a monotone submodular, set-dependent recovery benefit. A size-aware **Exact–Greedy** solver uses exact enumeration for small instances and marginal greedy selection for larger ones.
4. **Enforce the allocation.** Resolve logical checkpoints to physical recurrent components, revalidate their identities, and apply **recurrent-only reconciliation** while preserving attention-side cache management and native recovery semantics.

The main quantities are the planning target $T_p$, executable recovery frontier $E_p(S)$, and recovery gap $G_p(S)=T_p-E_p(S)$. The objective minimizes aggregate modeled recovery cost $J_t(S)$; equivalently, the allocator maximizes the recovery benefit $F_t(S)=J_t(\varnothing)-J_t(S)$ under the checkpoint budget.

## Quick start

Use Python 3.10 or later. From the repository root:

```bash
python3 examples/plan.py
python3 -m unittest tests.test_release_contract tests.test_strict_control -v
```

The example uses a small shared-prefix workflow and budget `K=2`. It selects `branch-a-deep` and `branch-b` using Exact Search and reports the selected set, per-continuation frontiers and gaps, modeled recovery cost, and CPU solving time.

| Continuation | $T$ | $E$ | $G=T-E$ |
| --- | ---: | ---: | ---: |
| `pending-a` | 8 | 6 | 2 |
| `pending-b` | 8 | 4 | 4 |

Try another budget:

```bash
python3 examples/plan.py --budget 1
```

For package installation and the full Python test suite:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[test]'
python3 -m pytest tests
```

The CPU regression tests also run through [GitHub Actions](.github/workflows/cpu.yml).

## Reproducing the evaluation

The evaluation covers **modeled checkpoint allocation, CPU solver performance, GPU request recovery, and runtime correctness**. The experimental workflow and input provenance are documented in the [experiment guide](docs/experiments.zh-CN.md) and [runtime integration guide](docs/runtime.zh-CN.md).

| Evaluation | Setup and measurement |
| --- | --- |
| Controlled allocation | Five shared-branch snapshots, four budgets (`K=1–4`), comparing Exact Search, Marginal Greedy, local search, and checkpoint-selection baselines |
| Trace-driven allocation | OpenHands and AgentX workflow snapshots with proportional and fixed checkpoint budgets; report selected checkpoints and modeled recovery costs |
| Controlled GPU recovery | Qwen3.5-9B on an NVIDIA H100 PCIe GPU (`TP=1`); measure request TTFT, historical prefill, and executable recovery frontiers |
| Trace-driven GPU recovery | Frozen OpenHands snapshots; evaluate per-request TTFT and recovery gaps across retention policies |
| Recovery-cost prediction | Position-aware cost fitting and held-out evaluation against gap-only prediction |
| Cross-architecture recovery | Kimi-Linear-48B-A3B-Instruct on two H100 PCIe GPUs (`TP=2`); validate MLA/KDA recovery behavior |

**Trace sources:** [OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) and [SemiAnalysis WekaTrace](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126), referred to as *AgentX* in the paper's derived workload analysis.

**Paper results.** In controlled four-request GPU experiments, RecurKeep reduces summed TTFT by **39.9%–65.0%** relative to five baselines. Over 24 frozen OpenHands snapshots, its mean per-request TTFT is **103.7 ms**. The position-aware model achieves **2.68% MAPE** on held-out recovery-cost configurations. The paper details the protocols and comparison baselines for each result.

### Result CSVs

The [`experiments/results/`](experiments/results/) directory organizes recorded evidence by experimental unit and source. See [`experiments/results/README.md`](experiments/results/README.md) for experiment groups, field definitions, source files, and aggregation rules.

| File | Content |
| --- | --- |
| [`gpu_recovery.csv`](experiments/results/gpu_recovery.csv) | Historical replay-length recovery measurements |
| [`recovery_frontiers.csv`](experiments/results/recovery_frontiers.csv) | Per-continuation planning targets, executable frontiers, and recovery gaps |
| [`cost_model.csv`](experiments/results/cost_model.csv) | Historical replay calibration and fitted values |
| [`checkpoint_allocation.csv`](experiments/results/checkpoint_allocation.csv) | Selected checkpoints and modeled allocation costs |
| [`solver_quality.csv`](experiments/results/solver_quality.csv) | Objective values and comparisons with Exact references |
| [`solver_runtime.csv`](experiments/results/solver_runtime.csv) | CPU solving measurements by policy and experiment group |
| [`runtime_validation.csv`](experiments/results/runtime_validation.csv) | Recorded runtime-state and output-correctness checks |

Use the `experiment_group` and `condition_id` columns to identify a condition; use `policy` for policy-level comparisons and `pending_id` for request-level records. The historical controlled study (`historical_strict20`) and renamed CPU study (`rename_cpu21`) contain overlapping conditions and are tracked as separate provenance groups.

The [artifact index](artifacts/README.zh-CN.md) documents the frozen inputs, archives, and source manifests used in result validation.

## SGLang runtime integration

RecurKeep's research prototype integrates with SGLang v0.5.17. The runtime path uses a **Strict Allocation Epoch** to observe pending requests, checkpoint candidates, compatibility relations, and component residency without modifying access metadata. After selecting the retained checkpoints, the controller revalidates physical bindings and calls `StateController.reconcile_strict` to apply recurrent-state protection and release.

Runtime integration preserves attention-side state management, native request recovery, and copy-on-write behavior. The [runtime guide](docs/runtime.zh-CN.md) describes state observation, bindings, reconciliation, and the SGLang adapter; the associated reference implementation and evaluation evidence are organized in `archives/runtime_reference.tar.gz`.

For the paper's implementation, Exact Search is configured with `n_max=12`, `m_max=4`, `K_max=4`, and an enumeration bound of `H_max=794`. These are solver dispatch parameters; the allocation formulation itself is expressed in terms of candidate count, pending count, and checkpoint budget.

## Repository layout

| Path | Purpose |
| --- | --- |
| `recurkeep/` | Recovery-cost model, allocation algorithms, validation, and state control |
| `recurkeep/adapters/` | SGLang integration interfaces |
| `examples/` | Runnable CPU allocation example |
| `tests/` | Allocator and runtime-control regression tests |
| `experiments/` | Frozen-instance readers, CPU experiment utilities, structured experimental results and CSV result exports|
| `artifacts/` | Frozen snapshots and artifact indexes |
| `archives/` | Archived runtime integration and experiment evidence |
| `validation/` | Reproducibility and regression comparisons |
| `docs/` | Experimental protocols, runtime integration |
| `tools/` | Export, validation, and release utilities |


