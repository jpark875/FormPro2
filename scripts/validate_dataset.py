"""Validate a reference dataset directory against the Phase 4 contract.

Run this before trusting a corpus. It applies exactly the checks ``load_corpus`` applies
at runtime, but reports every bad file instead of stopping at the first, so a set can be
fixed in one pass.

    python scripts/validate_dataset.py [data/reference] [--config configs/squat.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formpro.config import AppConfig  # noqa: E402
from formpro.dataset_loader import (  # noqa: E402
    DatasetError,
    ReferenceCorpus,
    load_sequence,
    summarize,
)
from formpro.schema import Phase  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pattern", default="*.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    config = AppConfig.load(args.config).dataset
    root = args.root or config.resolved_root()
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    paths = sorted(p for p in root.glob(args.pattern) if p.is_file())
    if not paths:
        print(f"no files matching {args.pattern!r} in {root}", file=sys.stderr)
        return 2

    good, failures = [], []
    for path in paths:
        try:
            good.append(load_sequence(path, config))
        except DatasetError as exc:
            failures.append(exc)

    if good:
        print(f"valid ({len(good)}/{len(paths)}):")
        print(summarize(good))
        corpus = ReferenceCorpus(tuple(good))
        print(f"\ncorpus: {corpus.coverage_report()}")

        low, high = corpus.ratio_span
        if high - low < config.min_ratio_span:
            print(
                f"\nwarning: femur_to_torso_ratio spans only {high - low:.3f}. The "
                f"build-adjusted tolerance band needs subjects across a wider range or "
                f"it behaves as a fixed threshold."
            )

        phases = Counter(p.value for s in good for p in s.phases)
        missing = [p.value for p in Phase if p.value not in phases]
        print("\nphase coverage: " + ", ".join(f"{k}={v}" for k, v in sorted(phases.items())))
        if missing:
            print(f"warning: no reference frames labelled: {', '.join(missing)}")

    if failures:
        print(f"\ninvalid ({len(failures)}/{len(paths)}):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("\nall files valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
