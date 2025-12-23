import sys
from pathlib import Path

# Add project root to path to ensure src can be imported if not installed as package
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import hydra
from omegaconf import DictConfig
from src.preprocessing import Preprocessor
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="preprocessing/default", version_base="1.3")
def main(cfg: DictConfig):
    log.info(f"Starting preprocessing for region: {cfg.region.name}")
    processor = Preprocessor(cfg)
    processor.run()

if __name__ == "__main__":
    main()
