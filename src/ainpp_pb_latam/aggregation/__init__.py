import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Score of a perfect forecast, used by the generic skill score.
PERFECT_SCORES = {
    "MAE": 0.0,
    "MSE": 0.0,
    "RMSE": 0.0,
    "CSI": 1.0,
    "ETS": 1.0,
    "HSS": 1.0,
    "POD": 1.0,
}


class Aggregator:
    @staticmethod
    def construct_tidy_dataframe(records: list) -> pd.DataFrame:
        """
        Converts a list of evaluation records into a tidy DataFrame.
        Each record should be a dict with: 'model', 'lead_time', 'threshold', 'metric_name', 'value'.
        """
        df = pd.DataFrame(records)
        return df

    @staticmethod
    def summarize(
        df: pd.DataFrame, group_by: list = ["model", "lead_time", "threshold", "metric_name"]
    ) -> pd.DataFrame:
        """
        Calculates mean and std across batches or regions.
        """
        if df.empty:
            return df

        summary_df = df.groupby(group_by, dropna=False)["value"].agg(["mean", "std"]).reset_index()
        return summary_df

    @staticmethod
    def combine(df_pooled: pd.DataFrame, df_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Merges pooled metrics (one value over the whole test set) with batch-mean metrics
        into a single table with columns: model, lead_time, threshold, metric_name,
        mean, std, aggregation.
        """
        frames = []
        if not df_pooled.empty:
            pooled = df_pooled.rename(columns={"value": "mean"})
            pooled["std"] = np.nan
            pooled["aggregation"] = "pooled"
            frames.append(pooled)
        if not df_batch.empty:
            batch = df_batch.copy()
            batch["aggregation"] = "batch_mean"
            frames.append(batch)
        if not frames:
            return pd.DataFrame()
        columns = ["model", "lead_time", "threshold", "metric_name", "mean", "std", "aggregation"]
        return pd.concat(frames, ignore_index=True)[columns]

    @staticmethod
    def skill_scores(df: pd.DataFrame, reference: str, value_col: str = "mean") -> pd.DataFrame:
        """
        Generic skill score of every model against a reference model (e.g. persistence):

            SS = (S - S_ref) / (S_perfect - S_ref)

        For error metrics (S_perfect = 0) this is ``1 - S / S_ref``. SS > 0 means the model
        beats the reference, SS = 1 is a perfect forecast. Only metrics listed in
        ``PERFECT_SCORES`` (or FSS_*) and pooled rows are used.
        """
        if "aggregation" in df.columns:
            df = df[df["aggregation"] == "pooled"]

        def perfect(metric: str) -> float:
            if metric.startswith("FSS_"):
                return 1.0
            return PERFECT_SCORES.get(metric, np.nan)

        df = df.assign(perfect=df["metric_name"].map(perfect)).dropna(subset=["perfect"])
        keys = ["lead_time", "threshold", "metric_name"]

        ref = df[df["model"] == reference][keys + [value_col]].rename(
            columns={value_col: "reference_value"}
        )
        if ref.empty:
            raise ValueError(f"Reference model '{reference}' not found in the results.")

        merged = df.merge(ref, on=keys, how="inner")
        denom = merged["perfect"] - merged["reference_value"]
        merged["skill"] = (merged[value_col] - merged["reference_value"]) / denom.where(denom != 0)
        merged["reference"] = reference
        return merged[["model", "reference"] + keys + [value_col, "reference_value", "skill"]]

    @staticmethod
    def save_results(df: pd.DataFrame, output_dir: str, filename: str = "evaluation_summary"):
        """
        Exports the Dataframe to CSV and Parquet.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        csv_path = out_path / f"{filename}.csv"
        parquet_path = out_path / f"{filename}.parquet"

        df.to_csv(csv_path, index=False)
        # using try block for parquet in case fastparquet or pyarrow is missing
        try:
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Results exported to {csv_path} and {parquet_path}")
        except ImportError:
            logger.info(
                f"Results exported to {csv_path} (parquet not saved, install pyarrow/fastparquet)"
            )
