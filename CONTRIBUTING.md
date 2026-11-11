# Contributing to RecurKeep

Start with the [README](README.md) for the project's scope and the CPU quick start. The [Chinese README](README.zh-CN.md) covers the same material.

## Development setup

Use Python 3.10 or newer. From the repository root, create a virtual environment and install the package with its test dependencies. On a POSIX shell:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[test]'
```

The CPU core uses only the Python standard library. Keep runtime-specific dependencies in the adapter or integration layer so that the basic example and tests remain usable without a model, GPU, SGLang, or external archives.

## Run checks for your change

For changes to planning, input validation, or state control, run the bundled example and basic contract tests:

```bash
python3 examples/plan.py
python3 -m unittest tests.test_release_contract tests.test_strict_control -v
```

Run the broader core regression suite after installing the test dependencies:

```bash
python3 -m pytest tests
```

These commands do not require the frozen research archives. The current [CPU CI workflow](.github/workflows/cpu.yml) runs the example and standard-library contract tests on Python 3.10 and 3.12; it does not run the full pytest suite or GPU experiments.

For changes affecting frozen experiment loading or historical result comparisons, prepare the archives as described in the [experiment guide](docs/experiments.md), then run:

```bash
python3 -m unittest experiments.test_frozen_reference -v
python3 -m experiments.compare --output results/cpu_comparison_review.csv
```

Choose a new output path for each comparison run. Report missing archives or unavailable runtime dependencies in the pull request instead of treating unexecuted checks as passed. Live integration changes need validation in a compatible environment; CPU tests alone do not establish GPU correctness.

For documentation-only changes, check local links, command accuracy, and agreement between translations. Reuse existing verification when commands and behavior are unchanged.

## Implementation and data conventions

- Keep changes focused and follow the surrounding code style. Add or update tests for changed behavior, especially failure paths.
- Preserve the strict separation between logical planning and physical execution. Unknown relationships, incomplete identities, and solver errors must not be hidden by greedy fallback.
- Preserve runtime isolation, apply-time freshness checks, and per-operation validation. Use the [runtime guide](docs/runtime.md) when changing integration behavior.
- Label synthetic fixtures and simulated runtime behavior explicitly. Do not present them as captured engine state, measured GPU performance, or historical research evidence.
- Treat frozen references and archived source as evidence. Explain any proposed change to their content or hashes, and keep its provenance inspectable.
- Keep generated results, local archives, model weights, and caches out of source changes. See `.gitignore` and the [provenance notes](docs/provenance.md).

## Documentation and translations

English is the default language for the public guides. Chinese counterparts use the `.zh-CN.md` suffix and link back to the English version. Update both versions when changing shared instructions, supported environments, or expected outputs. If a translation remains incomplete, state that in the pull request.

Keep commands, identifiers, snapshot names, hashes, and version constraints consistent across languages. Historical logs and archived source retain their original text; translation does not require rewriting evidence. Existing code comments and exception messages are not yet uniformly English. Changing an exception message may also require updating tests that match it.

## Pull request information

Describe the problem, the resulting behavior, and how you checked it. Include the commands you ran, their results, and any relevant limitations. For experiment changes, identify the input archive, environment, and whether results are modeled CPU costs or measured runtime values.

The original release manifests and check reports describe an earlier delivery; do not present them as verification of a new change. See [provenance and release records](docs/provenance.md) for their scope.

The project license is still pending author or rights-holder confirmation; [LICENSE_NOTICE.md](LICENSE_NOTICE.md) records its current status. Preserve source attribution for any third-party material.
