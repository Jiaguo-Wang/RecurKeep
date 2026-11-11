# Separate experiment archives


This directory records archive identities and provenance; large logs are kept outside the code repository. The original local delivery archives are stored in `../archives/`, which Git ignores. Public release requires uploading the archives to separate data storage and adding real download URLs here. No public download URLs are currently available.

- `frozen_cpu_snapshots.tar.gz`: five strict snapshots, 20 budget conditions, and the C/K3 fallback case. Includes normalized epochs, original snapshots, and frozen references.
- `runtime_reference.tar.gz`: unmodified research runtime hooks, dependency source, and C/K3 inputs. Historical names, hashes, and physical audit identities are preserved; physical identities cannot be reused across engines.

From the repository root, run `python3 tools/verify_release.py --archives-only` to verify both archive hashes before use. This check does not require extracted data or generated experiment results.

The minimal CPU example and basic tests do not require these archives. For archive extraction and historical rechecks, see the [experiment guide](../docs/experiments.md).
