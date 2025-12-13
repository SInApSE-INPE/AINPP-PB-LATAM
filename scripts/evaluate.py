import sys
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import NowcastingDataset
from src.evaluation.evaluator import Evaluator
from src.utils.standardization import LogZScoreStandardizer
from src.models.factory import get_model # Assuming this exists or using direct import if not

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Initialize Evaluation...")
    
    # 1. Dataset & Loader
    # Override dataset path if provided via env var or arg (hydra handles args)
    test_dataset = NowcastingDataset(cfg, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=False, 
        num_workers=cfg.dataset.num_workers
    )
    
    # 2. Model
    # Load model structure
    # Assuming we have a factory or direct init.
    # If get_model is not available, we need to import specific model class
    try:
        from src.models.factory import get_model
        model = get_model(cfg)
    except ImportError:
        # Fallback if factory doesn't exist, instantiate UNet directly (as seen in config)
        from src.models.unet import UNet # Hypothetical import based on conf
        # Or check conf/model/unet.yaml for class info
        pass 
        # For now, let's assume we can get the model. 
        # If verification fails, I'll fix this.
    
    # Check for checkpoint
    checkpoint_path = cfg.get("checkpoint", "outputs/checkpoint.pth") # Default or parameterized
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Evaluating initialized model.")

    # 3. Standardizer
    # Define params dir from config or default
    params_dir = cfg.get("evaluation", {}).get("params_dir", "/prj/ideeps/adriano.almeida/data/ainpp/legacy/model_params")
    region = cfg.get("evaluation", {}).get("region", "ainpp-amazon-basin")
    standardizer = LogZScoreStandardizer(params_dir=params_dir, region=region)
    
    # 4. Evaluator
    evaluator = Evaluator(
        model=model, 
        test_loader=test_loader, 
        config=cfg, 
        standardizer=standardizer,
        device=cfg.training.device if torch.cuda.is_available() else 'cpu'
    )
    
    # 5. Run
    evaluator.evaluate()

if __name__ == "__main__":
    main()
