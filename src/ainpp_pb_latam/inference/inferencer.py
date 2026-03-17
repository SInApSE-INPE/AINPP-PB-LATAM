import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import zarr
from omegaconf import DictConfig
from typing import Optional, Union, List, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Inferencer:
    """
    Unified module for inference in Nowcasting architectures (AINPP).
    Supports both prediction of an isolated sample (single) with customized saving, 
    and massive and optimized generation (historical) using Zarr Stores for high performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
    ):
        """
        Initializes the inference pipeline.
        
        Args:
            model (nn.Module): The pre-trained and loaded model (already in eval mode).
            config (DictConfig): Root configuration (used cfg.inference).
            device (torch.device): CUDA ou CPU.
        """
        self.model = model
        self.config = config.inference
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def infer_single(
        self,
        input_tensor: torch.Tensor,
        base_timestamp: str,
        coords: Optional[dict] = None
    ) -> Path:
        """
        Receives a single sample tensor and saves the prediction in a standardized structure.
        
        Args:
            input_tensor (torch.Tensor): O tensor de entrada, formato (1, Tin, C, H, W).
            base_timestamp (str): String com a data-hora inicial no formato YYYYMMDD_HHMM.
            coords (dict, opcional): Lats and Lons to inject into xarray (if applicable).
            
        Returns:
            Path: Caminho absoluto onde o arquivo foi salvo.
        """
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)  # Forces batch dimension
            
        input_tensor = input_tensor.to(self.device)
        logger.info(f"Inferindo amostra de tamanho: {input_tensor.shape}")
        
        # Predict
        prediction = self.model(input_tensor) # Saida: (B, Tout, C, H, W)
        prediction_np = prediction.squeeze().cpu().numpy() # Removes batch (Tout, C, H, W) or (Tout, H, W)
        
        # Mount directory in standard <ano>/<mes>/<dia>
        # base_timestamp ex: "20240115_1200"
        year, month, day = base_timestamp[:4], base_timestamp[4:6], base_timestamp[6:8]
        output_base = Path(self.config.single.output_dir) / year / month / day
        output_base.mkdir(parents=True, exist_ok=True)
        
        fmt = self.config.single.output_format
        filename = f"pred_gsmap_{base_timestamp}.{fmt}"
        output_file = output_base / filename
        
        # Format and save
        if fmt == "nc":
            # Xarray Dataset (NetCDF)
            t_len = prediction_np.shape[0] if prediction_np.ndim >= 3 else 1
            h, w = prediction_np.shape[-2], prediction_np.shape[-1]
            
            lats = coords.get("lat", np.arange(h)) if coords else np.arange(h)
            lons = coords.get("lon", np.arange(w)) if coords else np.arange(w)
            
            da = xr.DataArray(
                prediction_np,
                dims=["time", "lat", "lon"] if prediction_np.ndim == 3 else ["time", "channel", "lat", "lon"],
                name="precipitation"
            )
            ds = xr.Dataset({"precipitation": da})
            ds.to_netcdf(output_file)
        else:
            # PyTorch raw Tensor fallback
            torch.save(prediction.cpu(), output_file.with_suffix(".pt"))
            output_file = output_file.with_suffix(".pt")
            
        logger.info(f"Individual prediction saved in: {output_file}")
        return output_file

    @torch.no_grad()
    def infer_historical(
        self,
        dataloader: DataLoader,
    ) -> Path:
        """
        Performs inference over an entire dataset (Batch by Batch) and writes the result 
        diretamente para um disco Zarr (append), altamente otimizado para supercomputador.
        
        Args:
            dataloader (DataLoader): PyTorch Dataloader iterando na ordem do tempo.
            
        Returns:
            Path: O caminho para o Dataset Zarr salvo.
        """
        output_zarr = Path(self.config.historical.zarr_store)
        output_zarr.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Iniciando a Inferencia Historica para Dataset Zarr: {output_zarr}")
        
        root_group = None
        current_idx = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferring Historic")):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                # Some loaders return metadata as 3rd element
                # if it exists, useful for filling actual IDs.
            else:
                inputs = batch
            
            inputs = inputs.to(self.device)
            preds = self.model(inputs) # (B, Tout, C, H, W)
            preds_np = preds.cpu().numpy()
            
            batch_size = preds_np.shape[0]
            
            # Se for o primeiro batch, inicializa a estrutura do Zarr dinamicamente
            if root_group is None:
                # Optimal Chunk storage
                store = zarr.DirectoryStore(str(output_zarr))
                root_group = zarr.group(store=store, overwrite=True)
                
                # Chunk Optimization: Gets all the spatial scene at once, but breaks time
                c_sizes = (self.config.batch_size, preds_np.shape[1], preds_np.shape[2], preds_np.shape[3], preds_np.shape[4])
                
                # Dataset that can grow on Zero axis (Time/Batches)
                preds_arr = root_group.zeros(
                    'predictions', 
                    shape=(0, *preds_np.shape[1:]), 
                    chunks=c_sizes, 
                    dtype=np.float32,
                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
                )
                
            # Adiciona o batch atual via "append" otimizado
            root_group['predictions'].append(preds_np)
            current_idx += batch_size
            
        # Tries to consolidate metadata in zarr store for faster subsequent reads (Xarray loading)
        try:
            zarr.consolidate_metadata(str(output_zarr))
            logger.info("Metadados Zarr consolidados.")
        except Exception as e:
            logger.warning(f"Unable to consolidate Zarr metadata: {e}")
            
        logger.info(f"Historical inference complete! [{current_idx} samples saved in {output_zarr}]")
        return output_zarr
