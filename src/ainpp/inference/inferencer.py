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
    Módulo unificado para inferência nas arquiteturas de Nowcasting (AINPP).
    Suporta tanto a previsão de uma amostra isolada (single) com o salvamento customizado, 
    quanto a geração massiva e otimizada (historical) utilizando Zarr Stores para alta performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
    ):
        """
        Inicializa o pipeline de inferência.
        
        Args:
            model (nn.Module): O modelo pré-treinado e carregado (já em mode eval).
            config (DictConfig): Configuração raiz (usada cfg.inference).
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
        Recebe um tensor de amostra única e salva a previsão em estrutura padronizada.
        
        Args:
            input_tensor (torch.Tensor): O tensor de entrada, formato (1, Tin, C, H, W).
            base_timestamp (str): String com a data-hora inicial no formato YYYYMMDD_HHMM.
            coords (dict, opcional): Lats e Lons para injetar no xarray (caso aplicável).
            
        Returns:
            Path: Caminho absoluto onde o arquivo foi salvo.
        """
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)  # Força batch dimension
            
        input_tensor = input_tensor.to(self.device)
        logger.info(f"Inferindo amostra de tamanho: {input_tensor.shape}")
        
        # Predict
        prediction = self.model(input_tensor) # Saida: (B, Tout, C, H, W)
        prediction_np = prediction.squeeze().cpu().numpy() # Remove batch (Tout, C, H, W) ou (Tout, H, W)
        
        # Montar diretório no padrão <ano>/<mes>/<dia>
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
            
        logger.info(f"Previsão individual salva em: {output_file}")
        return output_file

    @torch.no_grad()
    def infer_historical(
        self,
        dataloader: DataLoader,
    ) -> Path:
        """
        Faz a inferência sobre um dataset inteiro (Batch por Batch) e escreve o resultado 
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
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferindo Histórico")):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                # Alguns loaders retornam metadados como 3o elemento
                # se existir, útil para preencher os IDs reais.
            else:
                inputs = batch
            
            inputs = inputs.to(self.device)
            preds = self.model(inputs) # (B, Tout, C, H, W)
            preds_np = preds.cpu().numpy()
            
            batch_size = preds_np.shape[0]
            
            # Se for o primeiro batch, inicializa a estrutura do Zarr dinamicamente
            if root_group is None:
                # Armazenamento em Chunk ótimo
                store = zarr.DirectoryStore(str(output_zarr))
                root_group = zarr.group(store=store, overwrite=True)
                
                # Otimização de Chunks: Pega todo a cena espacial de uma vez, mas quebra o tempo
                c_sizes = (self.config.batch_size, preds_np.shape[1], preds_np.shape[2], preds_np.shape[3], preds_np.shape[4])
                
                # Dataset que pode crescer no eixo Zero (Time/Batches)
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
            
        # Tenta consolidar o metadado no zarr store para leituras mais rapidas subsequentes (Xarray loading)
        try:
            zarr.consolidate_metadata(str(output_zarr))
            logger.info("Metadados Zarr consolidados.")
        except Exception as e:
            logger.warning(f"Não foi possível consolidar metadados Zarr: {e}")
            
        logger.info(f"Inferência histórica completa! [{current_idx} amostras salvas em {output_zarr}]")
        return output_zarr
