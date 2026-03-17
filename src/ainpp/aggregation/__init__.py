import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
    def summarize(df: pd.DataFrame, group_by: list = ['model', 'lead_time', 'threshold', 'metric_name']) -> pd.DataFrame:
        """
        Calculates mean and std across batches or regions.
        """
        if df.empty:
            return df
            
        summary_df = df.groupby(group_by, dropna=False)['value'].agg(['mean', 'std']).reset_index()
        return summary_df
        
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
            logger.info(f"Results exported to {csv_path} (parquet not saved, install pyarrow/fastparquet)")
