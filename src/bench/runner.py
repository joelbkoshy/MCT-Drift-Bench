"""Shared execution helper: resumable, incrementally written runs.

Generation on CPU is slow enough that losing a part-finished run to an
interruption is expensive, and rerunning only some cells would break the
fixed-seed guarantee. Rows are therefore appended as they are produced and
already-completed cells are skipped on restart.

Crash safety needs one thing beyond appending. A process killed mid-write, or a
machine that loses power, leaves a truncated final record: a row with too few
fields, or a response field whose opening quote is never closed. Left in place
that record both corrupts the file for any later reader and, because its key
fields are written first, marks its own cell as finished -- so the run would skip
regenerating it. Every file is therefore repaired before it is appended to.
"""

from __future__ import annotations

import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

# Generated responses are long; the default field limit can reject them.
csv.field_size_limit(10_000_000)


@contextmanager
def single_instance(lock_path: Path):
    """Refuse to start if another run already holds the lock.

    Two processes appending to the same results file interleave their rows and
    corrupt it, and the crash repair cannot tell that apart from a truncated
    write. An advisory OS lock is used rather than a PID file because the kernel
    releases it when the holder dies, so a machine that loses power does not
    leave a stale lock that has to be cleared by hand.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    # Lock byte 0 explicitly. A file opened for append starts positioned at EOF,
    # so two processes would otherwise lock different byte ranges and both
    # succeed -- the lock would look present and enforce nothing.
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                print(
                    f"another run is already in progress (lock: {lock_path}).\n"
                    "Check .\\status.ps1 -- if it is still advancing, let it finish.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                print(f"another run is already in progress (lock: {lock_path}).",
                      file=sys.stderr)
                raise SystemExit(2)

        yield
    finally:
        try:
            handle.close()
        except Exception:
            pass


class ResumableWriter:
    def __init__(self, path: Path, fieldnames: list[str], key: tuple[str, ...]):
        self.path = path
        self.fieldnames = fieldnames
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired = self._repair()
        self._done = self._load_done()
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._fh.flush()

    def _repair(self) -> int:
        """Drop any trailing incomplete record and normalise the file.

        Two distinct corruptions have to be handled. A cut between fields leaves
        a short row, which is easy to spot. A cut *inside* the final field is
        not: ``response`` is the last column, so an unterminated quote running to
        end-of-file still parses as a full-width row, and the next appended row
        is then swallowed into that open quoted field instead of becoming a row
        of its own. The file is therefore rewritten from the parsed rows on every
        open, which closes any dangling quote, and the final record is discarded
        whenever the file does not end on a clean record terminator.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0

        ends_clean = self.path.read_bytes().endswith(b"\n")

        with self.path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except (StopIteration, csv.Error):
                return 0

            if header != self.fieldnames:
                raise RuntimeError(
                    f"{self.path.name} was written with a different schema. "
                    "Move it aside before rerunning."
                )

            complete, dropped = [], 0
            try:
                for row in reader:
                    if len(row) == len(header):
                        complete.append(row)
                    else:
                        dropped += 1
            except csv.Error:
                dropped += 1

        if not ends_clean and complete:
            complete.pop()
            dropped += 1

        tmp = self.path.with_suffix(self.path.suffix + ".repair")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(complete)
        tmp.replace(self.path)

        if dropped:
            print(
                f"[resume] repaired {self.path.name}: dropped {dropped} "
                f"incomplete record(s), kept {len(complete)}",
                flush=True,
            )
        return dropped

    def _load_done(self) -> set[tuple]:
        if not self.path.exists():
            return set()
        with self.path.open(newline="", encoding="utf-8") as fh:
            return {
                tuple(row[k] for k in self.key)
                for row in csv.DictReader(fh)
                if all(row.get(k) for k in self.key)
            }

    def already_done(self, *values: str) -> bool:
        return tuple(str(v) for v in values) in self._done

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()
        self._done.add(tuple(str(row[k]) for k in self.key))

    def completed(self) -> int:
        return len(self._done)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def progress(label: str, done: int, total: int, extra: str = "") -> None:
    pct = 100.0 * done / total if total else 0.0
    print(f"[{label}] {done}/{total} ({pct:5.1f}%) {extra}", flush=True)


def flatten(rows: Iterable[dict]) -> list[dict]:
    return list(rows)
