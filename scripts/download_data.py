"""Download required data files from Kaggle.

Run this script once before your first forecast to populate data/processed/:

    python scripts/download_data.py

Prerequisites:
    1. Install the Kaggle CLI:  pip install kaggle
    2. Create a Kaggle API token at https://www.kaggle.com/settings -> API
       -> "Create New Token". This downloads kaggle.json.
    3. Place kaggle.json in:
         Windows: C:\\Users\\<YourUser>\\.kaggle\\kaggle.json
         macOS/Linux: ~/.kaggle/kaggle.json
       Or set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.

What this script downloads:
    - results.csv   (martj42/international-football-results)
      ~50 000 international matches from 1872 to present.
      Required columns: date, home_team, away_team, home_score, away_score,
                        tournament, city, country, neutral

    - rankings.csv  (cashncarry/fifaworldranking)  [optional]
      FIFA ranking snapshots for the RankingDiff feature.
      Only needed for the learned gradient-boosting model.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_EXTERNAL = REPO_ROOT / "data" / "external"


def _check_kaggle() -> None:
    if shutil.which("kaggle") is None:
        print("ERROR: 'kaggle' CLI not found. Run: pip install kaggle")
        sys.exit(1)


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def download_results() -> None:
    """Download and place the international football results CSV."""
    dest = DATA_PROCESSED / "results.csv"
    if dest.exists():
        print(f"✓ {dest} already exists — skipping download.")
        print("  Delete the file and re-run to force a fresh download.")
        return

    print("\n── Downloading international football results ──")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    _run(
        ["kaggle", "datasets", "download", "-d", "martj42/international-football-results",
         "--unzip", "-p", str(DATA_RAW)],
        cwd=REPO_ROOT,
    )

    # The Kaggle dataset unzips as results.csv in DATA_RAW.
    src = DATA_RAW / "results.csv"
    if not src.exists():
        print(f"ERROR: Expected {src} after unzip — check the Kaggle dataset structure.")
        sys.exit(1)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"✓ Copied {src} → {dest}")


def download_fifa_rankings() -> None:
    """Download the FIFA ranking time-series (optional, for RankingDiff feature)."""
    dest = DATA_EXTERNAL / "fifa_rankings.csv"
    if dest.exists():
        print(f"✓ {dest} already exists — skipping.")
        return

    print("\n── Downloading FIFA world rankings ──")
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    _run(
        ["kaggle", "datasets", "download", "-d", "cashncarry/fifaworldranking",
         "--unzip", "-p", str(DATA_RAW)],
        cwd=REPO_ROOT,
    )

    # The Kaggle dataset unzips a CSV (name may vary); find it.
    candidates = list(DATA_RAW.glob("*.csv"))
    ranking_files = [f for f in candidates if "rank" in f.name.lower()]
    if not ranking_files:
        ranking_files = candidates  # take the first CSV if name doesn't match
    if not ranking_files:
        print("WARNING: Could not find a CSV in the rankings download. Check data/raw/ manually.")
        return

    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ranking_files[0], dest)
    print(f"✓ Copied {ranking_files[0]} → {dest}")


def main() -> None:
    _check_kaggle()

    import argparse
    parser = argparse.ArgumentParser(description="Download WC2026 Predictor data from Kaggle.")
    parser.add_argument("--results-only", action="store_true",
                        help="Only download the results CSV (skip optional FIFA rankings).")
    args = parser.parse_args()

    download_results()

    if not args.results_only:
        download_fifa_rankings()

    print("\n✓ Done. Run 'wc2026 demo' to verify the install, then 'make predict' for a real forecast.")


if __name__ == "__main__":
    main()
