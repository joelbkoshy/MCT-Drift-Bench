"""Verify the single-instance lock rejects a concurrent run.

The results files are opened in append mode. Two processes appending interleave
their rows, and the crash repair cannot distinguish that from a truncated write,
so the guarantee that matters is that a second run never starts. This checks it
end to end with two real processes rather than by inspection.

    python tools/check_lock.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bench.runner import single_instance  # noqa: E402

LOCK = REPO / "results" / ".locktest"

CHILD = (
    "import sys;"
    f"sys.path.insert(0, r'{REPO}');"
    "from pathlib import Path;"
    "from src.bench.runner import single_instance;"
    f"ctx = single_instance(Path(r'{LOCK}'));"
    "ctx.__enter__();"
    "print('ACQUIRED')"
)


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.unlink(missing_ok=True)

    print("parent: acquiring lock")
    with single_instance(LOCK):
        child = subprocess.run(
            [sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60
        )
        blocked = child.returncode == 2 and "ACQUIRED" not in child.stdout
        print(f"  child exit code : {child.returncode} (expected 2)")
        print(f"  child stderr    : {child.stderr.strip() or '(none)'}")
        print(f"  BLOCKED WHILE HELD : {blocked}")

    time.sleep(0.2)
    print("parent: lock released")
    child = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60
    )
    reacquired = child.returncode == 0 and "ACQUIRED" in child.stdout
    print(f"  child exit code : {child.returncode} (expected 0)")
    print(f"  ACQUIRABLE ONCE FREE : {reacquired}")

    LOCK.unlink(missing_ok=True)
    ok = blocked and reacquired
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
