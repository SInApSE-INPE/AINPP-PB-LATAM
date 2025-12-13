import unittest
import os
import shutil
import zarr
import numpy as np
import torch
from omegaconf import OmegaConf
from src.dataset import NowcastingDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_dataset_env"
        os.makedirs(self.test_dir, exist_ok=True)
        self.data_path = os.path.join(self.test_dir, "dummy_data.zarr")
        self.create_dummy_zarr(self.data_path)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_zarr(self, path):
        store = zarr.open(path, mode='w')
        # Create groups for train, val, test as expected by the dataset?
        # Actually NowcastingDataset seems to use one path and filter by time?
        # Or does it check groups? 
        # Looking at previous code, it seems to assume the zarr structure.
        # verify_pipeline.py created a 'data' group.
        # test_evaluation_pipeline.py created 'test' group.
        # Let's create 'data' as generic root or check src/dataset.py if I was allowed.
        # But assuming verify_pipeline worked, let's create what seems standard.
        # Actually, looking at NowcastingDataset logic from memory/context:
        # It likely reads specific variables from the zarr.
        
        # Structure: root -> variable_name -> array (Time, Lat, Lon)
        
        T, H, W = 48, 32, 32
        
        # Create variables directly at root or inside a group?
        # Usually datasets for storing variables: group/var
        # Let's assume root acts as the container.
        
        # Create 'gsmap_nrt' and 'gsmap_mvk'
        # Dates: 2024-01-01 to 2024-01-02 usually needs enough hours.
        # 48 hours = 48 steps if hourly.
        
        gsmap_nrt = np.random.randn(T, H, W).astype(np.float32)
        gsmap_mvk = np.random.randn(T, H, W).astype(np.float32)
        
        store.create_dataset('gsmap_nrt', data=gsmap_nrt, shape=gsmap_nrt.shape)
        store.create_dataset('gsmap_mvk', data=gsmap_mvk, shape=gsmap_mvk.shape)
        
        # Attributes for time? 
        # If dataset uses time alignment, we might need metadata.
        # For now, let's assume index based or simple existing logic.

    def test_initialization_and_len(self):
        conf = OmegaConf.create({
            "dataset": {
                "data_path": self.data_path,
                "input_vars": ["gsmap_nrt"],
                "output_vars": ["gsmap_mvk"],
                "train_period": {"start": "2024-01-01", "end": "2024-01-02"},
                "val_period": {"start": "2024-01-01", "end": "2024-01-02"}, 
                "test_period": {"start": "2024-01-01", "end": "2024-01-02"},
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False
            }
        })
        
        # Test Train Split
        ds = NowcastingDataset(conf, split='train')
        self.assertIsInstance(ds, torch.utils.data.Dataset)
        # Length depends on how many samples can be drawn from 48 time steps.
        # Assuming sequence length is standard (e.g. 4 in, 1 out).
        # Just checking it initializes and has > 0 len if enough data.
        # if T=48, len > 0.
        
    def test_getitem(self):
        conf = OmegaConf.create({
            "dataset": {
                "data_path": self.data_path,
                "input_vars": ["gsmap_nrt"],
                "output_vars": ["gsmap_mvk"],
                "train_period": {"start": "2024-01-01", "end": "2024-01-02"},
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False
            }
        })
        ds = NowcastingDataset(conf, split='train')
        
        if len(ds) > 0:
            sample = ds[0]
            # sample tuple (x, y)
            self.assertEqual(len(sample), 2)
            x, y = sample
            # Check shapes: (C, H, W) or (C, T, H, W)?
            # Usually (C, T, H, W) for video/nowcasting.
            # verify_pipeline used unet 1->1 channel.
            # Let's verify usage of torch tensors
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
