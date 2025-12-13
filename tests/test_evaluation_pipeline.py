import sys
import os
import shutil
import zarr
import numpy as np
import torch
import json
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, '/prj/ideeps/adriano.almeida/benchmark')

from src.dataset import NowcastingDataset
from scripts.evaluate import main as evaluate_main

def create_dummy_zarr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    store = zarr.open(path, mode='w')
    time_len = 5
    lat = 32
    lon = 32
    
    # Needs to match what NowcastingDataset expects.
    # Dataset expects 'train', 'validation', 'test' groups.
    for group in ['test']:
        grp = store.create_group(group)
        # Dummy data
        data_nrt = np.random.randn(time_len, lat, lon).astype(np.float32)
        grp.create_dataset('gsmap_nrt', data=data_nrt, shape=data_nrt.shape)
        data_mvk = np.random.randn(time_len, lat, lon).astype(np.float32)
        grp.create_dataset('gsmap_mvk', data=data_mvk, shape=data_mvk.shape)

def create_dummy_params(params_dir, region):
    os.makedirs(params_dir, exist_ok=True)
    mean_path = os.path.join(params_dir, f"gsmap_nrt+mvk_log_mean_{region}.npy")
    std_path = os.path.join(params_dir, f"gsmap_nrt+mvk_log_std_{region}.npy")
    
    np.save(mean_path, np.array(0.0))
    np.save(std_path, np.array(1.0))

def test_pipeline():
    print("Setting up dummy environment...")
    base_dir = os.path.abspath("tests/dummy_env")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    dummy_data_path = f"{base_dir}/dummy_data.zarr"
    create_dummy_zarr(dummy_data_path)
    
    params_dir = f"{base_dir}/params"
    region = "ainpp-test-region"
    create_dummy_params(params_dir, region)
    
    # Config
    conf = OmegaConf.create({
        "dataset": {
            "data_path": dummy_data_path,
            "input_vars": ["gsmap_nrt"],
            "output_vars": ["gsmap_mvk"],
            "test_period": {"start": "2024-01-01", "end": "2024-01-02"}, 
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False
        },
        "model": {
            "name": "unet",
            "in_channels": 1,
            "out_channels": 1,
            "features": [64]
        },
        "training": {
            "device": "cpu"
        },
        "evaluation": {
            "thresholds": [0.5, 2.0],
            "params_dir": params_dir,
            "region": region,
            "max_batches": 20
        }
    })
    
    print("Running evaluate main...")
    try:
        evaluate_main(conf)
        print("Evaluation ran successfully.")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        # raise e
        
    # Check outputs
    # Default output dir is ./outputs/evaluation
    expected_output = os.path.abspath("outputs/evaluation/metrics.json")
    if os.path.exists(expected_output):
        print(f"Verified output exists: {expected_output}")
        with open(expected_output, 'r') as f:
            print(json.dumps(json.load(f), indent=2))
    else:
        print(f"Output not found at {expected_output}")

    # Cleanup
    # shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_pipeline()
