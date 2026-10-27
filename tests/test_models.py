from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ainpp_pb_latam.models.afno.forecaster import AFNO2D

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def _compose(overrides=()):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


class TestConfigDefaults:
    def test_default_training_is_supervised(self) -> None:
        cfg = _compose()
        assert cfg.training.mode == "supervised"

    def test_afno_img_size_follows_dataset_patch(self) -> None:
        cfg = _compose(
            [
                "model=afno/direct",
                "dataset.dataset.patch_height=160",
                "dataset.dataset.patch_width=200",
            ]
        )
        assert OmegaConf.to_container(cfg.model, resolve=True)["img_size"] == [160, 200]

    @pytest.mark.parametrize("model", ["unet/autoregressive", "unet/direct", "afno/direct"])
    def test_models_follow_global_timesteps(self, model: str) -> None:
        cfg = _compose([f"model={model}", "input_timesteps=8", "output_timesteps=3"])
        resolved = OmegaConf.to_container(cfg.model, resolve=True)
        assert resolved["input_timesteps"] == 8
        assert resolved["output_timesteps"] == 3


class TestAFNO:
    def _model(self, **kwargs) -> AFNO2D:
        params = dict(
            img_size=(40, 30),
            input_timesteps=4,
            output_timesteps=2,
            embed_dim=16,
            depth=1,
            patch_size=10,
            num_blocks=2,
        )
        params.update(kwargs)
        return AFNO2D(**params)

    def test_forward_shape_and_nonnegative(self) -> None:
        y = self._model()(torch.randn(2, 4, 1, 40, 30))
        assert y.shape == (2, 2, 1, 40, 30)
        assert (y >= 0).all()

    def test_nonnegativity_none_allows_negative(self) -> None:
        torch.manual_seed(0)
        y = self._model(nonnegativity="none")(torch.randn(2, 4, 1, 40, 30))
        assert (y < 0).any()

    def test_wrong_input_size_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="img_size"):
            self._model()(torch.randn(1, 4, 1, 30, 30))

    def test_patch_size_must_divide_img_size(self) -> None:
        with pytest.raises(ValueError, match="patch_size"):
            self._model(img_size=(45, 30))


class TestTrainGuard:
    def test_train_task_rejects_gan_profile(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ainpp_main", Path(CONF_DIR).parent / "main.py"
        )
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)

        with pytest.raises(ValueError, match="training=default"):
            main._run_train(_compose(["training=gan"]))
