import hydra
from omegaconf import DictConfig
from src.evaluation import Evaluator

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Starting evaluation...")
    evaluator = Evaluator(cfg)
    # evaluator.evaluate(...)

if __name__ == "__main__":
    main()
