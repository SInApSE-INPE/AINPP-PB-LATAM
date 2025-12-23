import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import pandas as pd

class NowcastingDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data_path = config.dataset.data_path
        
        # Open Zarr group
        # Assuming the zarr file has a structure compatible with xarray or just arrays
        # For this implementation, I'll assume it's a group with variables as arrays
        # and a time coordinate.
        try:
            self.store = zarr.open(self.data_path, mode='r')
        except Exception as e:
            print(f"Warning: Could not open Zarr file at {self.data_path}. Using dummy data for initialization.")
            self.store = None

        # Define time periods
        if split == 'train':
            period = config.dataset.train_period
        elif split == 'val':
            period = config.dataset.val_period
        elif split == 'test':
            period = config.dataset.test_period
        else:
            raise ValueError(f"Unknown split: {split}")

        self.start_date = pd.to_datetime(period.start)
        self.end_date = pd.to_datetime(period.end)
        
        if self.store is not None:
             # Assuming 'time' coordinate exists in zarr
             if 'time' in self.store:
                 try:
                     # Load time array.
                     time_array = self.store['time'][:]
                     
                     # Check for CF conventions in attributes
                     if hasattr(self.store['time'], 'attrs') and 'units' in self.store['time'].attrs:
                         units_str = self.store['time'].attrs['units']
                         # Format: "unit since date"
                         try:
                             parts = units_str.split(' since ')
                             if len(parts) == 2:
                                 unit = parts[0]
                                 origin = parts[1]
                                 
                                 # Map common CF units to pandas units
                                 # CF: hours, minutes, seconds, days
                                 # Pandas: h, m, s, D
                                 unit_map = {'hours': 'h', 'minutes': 'm', 'seconds': 's', 'days': 'D'}
                                 pd_unit = unit_map.get(unit, 'h') # Default to hours if unknown/complex
                                 
                                 all_times = pd.to_datetime(time_array, unit=pd_unit, origin=pd.Timestamp(origin))
                             else:
                                 # Fallback
                                 all_times = pd.to_datetime(time_array)
                         except Exception as e:
                             print(f"Error parsing time units '{units_str}': {e}. Fallback to default.")
                             all_times = pd.to_datetime(time_array)
                     else:
                         all_times = pd.to_datetime(time_array)
                     
                     # Create boolean mask
                     mask = (all_times >= self.start_date) & (all_times <= self.end_date)
                     
                     # Get valid indices
                     self.valid_indices = np.where(mask)[0]
                     self.length = len(self.valid_indices)
                     
                     if self.length == 0:
                         print(f"Warning: No samples found in period {self.start_date} to {self.end_date}. Data range: {all_times.min()} to {all_times.max()}")
                         
                 except Exception as e:
                     print(f"Error loading/filtering time: {e}. Using dummy length.")
                     self.valid_indices = np.arange(100)
                     self.length = 100
             else:
                 print("Warning: 'time' coordinate not found in Zarr. Using dummy length.")
                 self.valid_indices = np.arange(100)
                 self.length = 100
        else:
             # Dummy mode
             self.valid_indices = np.arange(100)
             self.length = 100
        
        self.input_vars = config.dataset.input_vars
        self.output_vars = config.dataset.output_vars

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Placeholder for data loading logic
        # In a real scenario, we would map idx to a timestamp and load the corresponding slice
        
        # Dummy data generation for testing the pipeline
        # Assuming (C, H, W)
        C_in = len(self.input_vars)
        C_out = len(self.output_vars)
        H, W = 256, 256 # Dummy spatial dims
        
        input_data = torch.randn(C_in, H, W)
        target_data = torch.randn(C_out, H, W)
        
        return input_data, target_data
