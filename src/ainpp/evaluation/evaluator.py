import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ainpp.metrics.continuous import ContinuousMetrics
from ainpp.metrics.categorical import CategoricalMetrics
from ainpp.metrics.probabilistic import ProbabilisticMetrics
from ainpp.metrics.object_based import ObjectBasedMetrics
from ainpp.metrics.sharpness import SharpnessMetrics
from ainpp.metrics.consistency import ConsistencyMetrics
from ainpp.aggregation import Aggregator

class Evaluator:
    """
    Orchestrates the evaluation pipeline scaling through multiple thresholds, 
    lead times, and compute all mathematical metrics individually.
    """
    def __init__(self, model, test_loader, config, standardizer=None, device='cpu'):
        self.model = model
        self.loader = test_loader
        self.config = config
        self.standardizer = standardizer
        self.device = device
        
        # Protocol
        eval_cfg = config.get("evaluation", {})
        
        # Directories
        out_cfg = eval_cfg.get("output_dir", "outputs/evaluation")
        self.base_output_dir = Path(out_cfg)
        self.data_dir = self.base_output_dir / "data"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.thresholds = eval_cfg.get("thresholds_mm_h", [0.1, 1.0, 5.0, 10.0])
        self.lead_times = eval_cfg.get("lead_times_min", [10, 20, 30, 40, 50, 60])
        
        # What to compute
        self.compute_cat = eval_cfg.get("categorical", True)
        self.compute_cont = eval_cfg.get("continuous", True)
        self.compute_prob = eval_cfg.get("probabilistic", True)
        self.compute_obj = eval_cfg.get("object_based", True)
        self.compute_sharp = eval_cfg.get("sharpness", True)
        self.compute_consist = eval_cfg.get("consistency", True)

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        
        records = []
        model_name = self.config.model.get("_target_", "Unknown").split('.')[-2]
        
        print(f"Starting Benchmark Evaluation for model {model_name}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.loader, desc="Evaluating Lead Times & Thresholds")):
                if isinstance(batch, (list, tuple)):
                    data, target = batch[0], batch[1]
                else: 
                    data, target = batch, batch
                    
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Inferencia Direta ou Autorregressiva encapsulada no modelo
                output = self.model(data) # [B, T, C, H, W] ou similar
                
                # Formato esperado: (B, LeadTimes, H, W)
                if output.ndim == 5:
                    output = output.squeeze(2)
                    target = target.squeeze(2)
                    
                output_np = output.cpu().numpy()
                target_np = target.cpu().numpy()
                
                # Iterar sobre os horizontes de previsão (Lead Times)
                T = output_np.shape[1]
                for t in range(T):
                    lead_name = f"T+{t+1}" # ou conversao real para minutos se configurado
                    
                    pred_t = output_np[:, t, ...]
                    target_t = target_np[:, t, ...]
                    
                    # 1. Metricas Continuas (nao dependem de limiar)
                    if self.compute_cont:
                        cont_metrics = ContinuousMetrics.compute(pred_t, target_t)
                        for m_name, val in cont_metrics.items():
                            records.append({
                                "model": model_name, "lead_time": lead_name,
                                "threshold": np.nan, "metric_name": m_name, "value": val
                            })
                            
                    # 2. Sharpness & Consistency (nao dependem de limiar)
                    if self.compute_sharp:
                        sharp_metrics = SharpnessMetrics.compute(pred_t, target_t)
                        for m_name, val in sharp_metrics.items():
                            records.append({
                                "model": model_name, "lead_time": lead_name,
                                "threshold": np.nan, "metric_name": m_name, "value": val
                            })
                            
                    if self.compute_consist:
                        cons_metrics = ConsistencyMetrics.compute(pred_t, target_t)
                        for m_name, val in cons_metrics.items():
                            records.append({
                                "model": model_name, "lead_time": lead_name,
                                "threshold": np.nan, "metric_name": m_name, "value": val
                            })

                    # O que depende de Threshold
                    for thresh in self.thresholds:
                        if self.compute_cat:
                            cat_metrics = CategoricalMetrics.compute(pred_t, target_t, threshold=thresh)
                            for m_name, val in cat_metrics.items():
                                records.append({
                                    "model": model_name, "lead_time": lead_name,
                                    "threshold": thresh, "metric_name": m_name, "value": val
                                })
                                
                        if self.compute_prob:
                            prob_metrics = ProbabilisticMetrics.compute(pred_t, target_t, threshold=thresh)
                            for m_name, val in prob_metrics.items():
                                records.append({
                                    "model": model_name, "lead_time": lead_name,
                                    "threshold": thresh, "metric_name": m_name, "value": val
                                })
                                
                        if self.compute_obj:
                            obj_metrics = ObjectBasedMetrics.compute(pred_t, target_t, threshold=thresh)
                            for m_name, val in obj_metrics.items():
                                records.append({
                                    "model": model_name, "lead_time": lead_name,
                                    "threshold": thresh, "metric_name": m_name, "value": val
                                })
                                
        # Agregação Final
        print("Agregando métricas e salvando tabela Tidy...")
        df = Aggregator.construct_tidy_dataframe(records)
        df_summary = Aggregator.summarize(df)
        Aggregator.save_results(df_summary, output_dir=str(self.base_output_dir))
        
        return df_summary
