# Synthetic CPU example data

`demo.json` is a small bundled logical fixture with four candidate checkpoints, two pending requests, and a complete compatibility matrix. Its token IDs are manually chosen integers, and all physical admission conditions are simulated assumptions. It is not a runtime capture and does not validate paper results or GPU correctness.

The two requests share a prefix of length 2, then branch. With a budget of 2, the optimal retained set is `branch-a-deep` and `branch-b`. Their recovery frontiers are 6 and 4, and the remaining recovery lengths are 2 and 4, respectively. `manifest.json` records the file hash and available budgets.

From the repository root, run `python3 examples/plan.py`. For real frozen snapshots and historical rechecks, see the [experiment guide](../../docs/experiments.md).
