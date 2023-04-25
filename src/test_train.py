import pyrootutils
from omegaconf import DictConfig
import hydra
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning.pytorch as pl
import sys

sys.path.append("src/data")
sys.path.append("src/models")

if __name__ == "__main__":

    # set up python path
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")

    # func to test training
    def test_training_model(cfg: DictConfig):
        # create model
        model = hydra.utils.instantiate(cfg.model)

        # create datamodule
        datamodule = hydra.utils.instantiate(cfg.data)

        # # create logger
        # wandb_logger = WandbLogger(project="facial_landmarks")

        # create Trainer
        trainer = pl.Trainer(fast_dev_run=True)

        # training
        trainer.fit(model, datamodule)

    # def main
    @hydra.main(version_base="1.3", config_path=config_path, config_name="test_train")
    def main(cfg: DictConfig):
        test_training_model(cfg)

    # call main
    main()