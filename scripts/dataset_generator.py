"""Generate synthetic reference profiles across a range of builds.

Takes one verified perfect-form recording and re-expresses it as the same rep performed
by lifters of different femur-to-torso ratios, widening the corpus so the analyzer
interpolates far more often than it extrapolates.

    python scripts/dataset_generator.py data/reference/subject_1p00.json
    python scripts/dataset_generator.py source.json --ratios 0.7 0.9 1.1 --out data/reference

The biomechanical model lives in ``formpro/synthesis.py`` so it can be unit-tested; this
file is the command-line front end. Read that module's docstring before trusting the
output: the profiles it writes encode a modelling assumption, not an observed lifter, and
they are marked ``reference_optimal_synthetic`` so they stay distinguishable from real
recordings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formpro.config import AppConfig  # noqa: E402
from formpro.dataset_loader import DatasetError, load_sequence  # noqa: E402
from formpro.schema import FormLabel  # noqa: E402
from formpro.synthesis import (  # noqa: E402
    DEFAULT_TARGET_RATIOS,
    build_document,
    warp_frames,
)

log = logging.getLogger("dataset_generator")


def slug(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize reference profiles for uncovered builds.",
    )
    parser.add_argument("source", type=Path,
                        help="a verified perfect-form reference JSON file")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: the source's directory)")
    parser.add_argument("--ratios", type=float, nargs="+", default=None,
                        help=f"target ratios (default: {list(DEFAULT_TARGET_RATIOS)})")
    parser.add_argument("--prefix", default="synthetic",
                        help="output filename prefix (default: synthetic)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing generated files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset_config = AppConfig.load(args.config).dataset

    try:
        source = load_sequence(args.source, dataset_config)
    except DatasetError as exc:
        print(f"cannot read the source recording: {exc}", file=sys.stderr)
        return 2

    labels = {frame.form_label for frame in source.frames}
    if labels != {FormLabel.OPTIMAL}:
        # Warping an error rep would manufacture a whole family of wrong references.
        others = sorted(label.value for label in labels if label is not FormLabel.OPTIMAL)
        print(
            f"refusing to warp {args.source.name}: every frame must be labelled "
            f"optimal_form, but it also contains {others}. Synthesizing from a flawed rep "
            f"would propagate that flaw across every generated build.",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out or args.source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = args.ratios if args.ratios is not None else list(DEFAULT_TARGET_RATIOS)

    source_ratio = source.femur_to_torso_ratio
    k = source.proportions.tibia_to_femur_ratio
    print(
        f"source: {args.source.name}  {len(source)} frames  "
        f"femur/torso {source_ratio:.2f}  tibia/femur {k:.2f}"
    )

    written, skipped = 0, 0
    for target in sorted(ratios):
        destination = out_dir / f"{args.prefix}_{slug(target)}.json"
        if destination.exists() and not args.force:
            print(f"  {destination.name}: exists, skipping (use --force)")
            skipped += 1
            continue

        frames, report = warp_frames(source.frames, source_ratio, target, k)
        document = build_document(
            frames=frames,
            target_ratio=target,
            tibia_to_femur_ratio=k,
            source_name=args.source.name,
            source_ratio=source_ratio,
            camera_angle=source.metadata.camera_angle,
            exercise=source.metadata.exercise,
            fps_target=source.metadata.fps_target,
        )
        destination.write_text(json.dumps(document, indent=1), encoding="utf-8")

        # Read it straight back through the Phase 4 loader. A generator that emits files
        # the loader rejects is worse than no generator, and this catches it at write
        # time rather than at the start of a training session.
        try:
            load_sequence(destination, dataset_config)
        except DatasetError as exc:
            destination.unlink(missing_ok=True)
            print(f"  {destination.name}: FAILED validation, removed: {exc}",
                  file=sys.stderr)
            return 1

        notes = []
        if report.clamped_frames:
            notes.append(f"{report.clamped_frames} clamped")
        if report.infeasible_frames:
            notes.append(f"{report.infeasible_frames} infeasible")
        suffix = f"  [{', '.join(notes)}]" if notes else ""
        print(
            f"  {destination.name}: ratio {target:.2f}, lean shift "
            f"{report.mean_back_shift_deg:+.1f} deg mean / "
            f"{report.max_back_shift_deg:.1f} deg max{suffix}"
        )
        written += 1

    print(f"\nwrote {written} profile(s), skipped {skipped}")
    if written:
        print(
            "These are synthetic. They encode the midfoot-balance model, not measured "
            "lifters, and should be replaced with real recordings as those arrive.\n"
            "Verify the corpus with: python scripts/validate_dataset.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
