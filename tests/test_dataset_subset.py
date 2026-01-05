import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from src.datasets import NowcastingDataset

class TestDatasetSubset(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create({
            "dataset": {
                "data_path": "dummy.zarr",
                "train_period": {"start": "2020-01-01", "end": "2020-01-10"},
                "val_period": {"start": "2020-01-11", "end": "2020-01-20"},
                "input_vars": ["var1"],
                "output_vars": ["var2"]
            }
        })

    @patch('src.dataset.zarr')
    def test_time_filtering(self, mock_zarr):
        # Mock Zarr store
        mock_store = MagicMock()
        mock_zarr.open.return_value = mock_store
        
        # Create a time array covering 20 days
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        # Store expects index access returning value
        # mock_store['time'][:] should return the dates (or strings/int, but pd.to_datetime handles them)
        mock_store.__contains__.side_effect = lambda x: x == 'time'
        mock_store['time'].__getitem__.return_value = dates.values
        
        # Initialize dataset with 'train' split (2020-01-01 to 2020-01-10)
        # Should contain 10 days.
        ds_train = NowcastingDataset(self.config, split='train')
        
        # Verify length
        self.assertEqual(len(ds_train), 10)
        # Verify indices: first 10
        np.testing.assert_array_equal(ds_train.valid_indices, np.arange(10))

    @patch('src.dataset.zarr')
    def test_val_time_filtering(self, mock_zarr):
        mock_store = MagicMock()
        mock_zarr.open.return_value = mock_store
        
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        mock_store.__contains__.side_effect = lambda x: x == 'time'
        mock_store['time'].__getitem__.return_value = dates.values
        
        # Val split: 2020-01-11 to 2020-01-20
        # Indices 10 to 19 (10 values)
        ds_val = NowcastingDataset(self.config, split='val')
        
        self.assertEqual(len(ds_val), 10)
        np.testing.assert_array_equal(ds_val.valid_indices, np.arange(10, 20))
        
    @patch('src.dataset.zarr')
    def test_empty_period(self, mock_zarr):
        mock_store = MagicMock()
        mock_zarr.open.return_value = mock_store
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        mock_store.__contains__.side_effect = lambda x: x == 'time'
        mock_store['time'].__getitem__.return_value = dates.values
        
        # Period outside range
        self.config.dataset.train_period.start = "2021-01-01"
        self.config.dataset.train_period.end = "2021-01-01"
        
        ds = NowcastingDataset(self.config, split='train')
        self.assertEqual(len(ds), 0)

if __name__ == '__main__':
    unittest.main()
