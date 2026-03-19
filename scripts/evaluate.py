import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf


from ainpp_pb_latam.datasets import NowcastingDataset
from ainpp_pb_latam.evaluation.evaluator import Evaluator
from ainpp_pb_latam._utils.standardization import LogZScoreStandardizer
from hydra.utils import instantiate


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Initialize Evaluation...")

    # 1. Dataset & Loader
    # Override dataset path if provided via env var or arg (hydra handles args)
    test_dataset = NowcastingDataset(cfg, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    # 2. Model
    # Load model structure using Hydra
    model = instantiate(cfg.model)

    # Check for checkpoint
    checkpoint_path = cfg.get("checkpoint", "outputs/checkpoint.pth")  # Default or parameterized
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Evaluating initialized model.")

    # 3. Standardizer
    # Define params dir from config or default
    params_dir = cfg.get("evaluation", {}).get(
        "params_dir", "/prj/ideeps/adriano.almeida/data/ainpp/legacy/model_params"
    )
    region = cfg.get("evaluation", {}).get("region", "ainpp-amazon-basin")
    standardizer = LogZScoreStandardizer(params_dir=params_dir, region=region)

    # 4. Evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=cfg,
        standardizer=standardizer,
        device=cfg.training.device if torch.cuda.is_available() else "cpu",
    )

    # 5. Run
    evaluator.evaluate()


if __name__ == "__main__":
    main()
