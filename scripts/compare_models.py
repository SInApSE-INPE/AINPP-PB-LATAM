"""
Combine evaluation summaries of several models and compute skill scores against a reference.

Example:
    python scripts/compare_models.py \
        outputs/eval_persistence/evaluation_summary.csv \
        outputs/eval_unet/evaluation_summary.csv \
        --reference persistence/direct --output outputs/comparison
"""

import argparse
from pathlib import Path

import pandas as pd

from ainpp_pb_latam.aggregation import Aggregator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("summaries", nargs="+", help="evaluation_summary.csv files")
    parser.add_argument("--reference", required=True, help="Model name used as reference")
    parser.add_argument("--output", default="outputs/comparison", help="Output directory")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(path) for path in args.summaries], ignore_index=True)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    Aggregator.save_results(df, output_dir=str(out_dir), filename="all_models_summary")
    skill = Aggregator.skill_scores(df, reference=args.reference)
    Aggregator.save_results(skill, output_dir=str(out_dir), filename="skill_scores")
    print(f"Saved {len(skill)} skill rows to {out_dir / 'skill_scores.csv'}")


if __name__ == "__main__":
    main()
