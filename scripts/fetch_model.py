"""Download the BlazePose ``.task`` model bundle.

The Tasks API loads weights from a local file, and the bundles are 5–30 MB, so they are
fetched on demand and git-ignored rather than committed.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formpro.config import PROJECT_ROOT  # noqa: E402
from formpro.pose_estimator import MODEL_VARIANTS  # noqa: E402

BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
URLS = {
    v: f"{BASE_URL}/pose_landmarker_{v}/float16/1/pose_landmarker_{v}.task"
    for v in MODEL_VARIANTS
}


def download(variant: str, dest_dir: Path, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"pose_landmarker_{variant}.task"
    if dest.exists() and not force:
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = URLS[variant]
    print(f"downloading {url}")
    tmp = dest.with_suffix(".task.part")

    last = [0]

    def progress(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(min(100.0, block * block_size * 100.0 / total))
        if pct >= last[0] + 10:  # coarse, so piped logs stay readable
            last[0] = pct
            print(f"  {pct:3d}%", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    tmp.replace(dest)  # atomic swap so an interrupted download never looks complete
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODEL_VARIANTS, default="heavy",
                        help="heavy is the most accurate in Z; lite is fastest (default: heavy)")
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    args = parser.parse_args()

    try:
        download(args.variant, args.dir, args.force)
    except OSError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
