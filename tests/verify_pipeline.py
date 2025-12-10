import sys
import os
import shutil
import zarr
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import NowcastingDataset
from src.models.factory import get_model
from src.trainer import Trainer

def create_dummy_zarr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    store = zarr.open(path, mode='w')
    # Create dummy variables
    # Time, Lat, Lon
    # Assuming (Time, Lat, Lon) for simplicity in this dummy gen, 
    # but Dataset implementation assumes (C, H, W) output from __getitem__
    # The Dataset implementation actually generates random data in __getitem__ 
    # so the zarr content doesn't matter much for the *current* dummy implementation
    # but we should create it to avoid file not found errors.
    
    store.create_group('data')

def test_pipeline():
    print("Setting up dummy environment...")
    dummy_data_path = "tests/dummy_data.zarr"
    create_dummy_zarr(dummy_data_path)
    
    # Create a config object manually or load it
    conf = OmegaConf.create({
        "dataset": {
            "data_path": dummy_data_path,
            "input_vars": ["gsmap_nrt"],
            "output_vars": ["gsmap_mvk"],
            "train_period": {"start": "2018-01-01", "end": "2018-01-02"},
            "val_period": {"start": "2023-01-01", "end": "2023-01-02"},
            "test_period": {"start": "2024-01-01", "end": "2024-01-02"},
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False
        },
        "model": {
            "name": "unet",
            "in_channels": 1,
            "out_channels": 1
        },
        "training": {
            "epochs": 1,
            "learning_rate": 1e-3,
            "device": "cpu",
            "experiment_name": "test_run",
            "distributed": False,
            "dist_backend": "nccl"
        }
    })
    
    print("Initializing Dataset...")
    train_dataset = NowcastingDataset(conf, split='train')
    val_dataset = NowcastingDataset(conf, split='val')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
    
    print("Initializing Model...")
    model = get_model(conf)
    
    print("Initializing Trainer...")
    trainer = Trainer(model, train_loader, val_loader, conf)
    
    print("Starting Training Loop (Dry Run)...")
    trainer.fit()
    
    print("Verification Successful!")
    
    # Cleanup
    if os.path.exists(dummy_data_path):
        shutil.rmtree(dummy_data_path)

if __name__ == "__main__":
    test_pipeline()
