from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import pandas as pd
import numpy as np


class DataModule1(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.3, 0),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data9 = pd.read_excel("data/Dataset1/data_match9.xlsx", engine='openpyxl')
            data10 = pd.read_excel("data/Dataset1/data_match10.xlsx", engine='openpyxl')

            data9.dropna(inplace=True)
            data10.dropna(inplace=True)

            X_train_9 = data9[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']].values
            y_train_9 = data9['value'].values

            X_train_10 = data10[['B04B','B05B','B06B','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB','CAPE','TCC','TCW','TCWV']].values
            y_train_10 = data10['value'].values

            X_train = np.concatenate((X_train_9, X_train_10), axis=0)
            y_train = np.concatenate((y_train_9, y_train_10), axis=0)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

            dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = self.data_val


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from tqdm import tqdm

    def test_datamodule(cfg: DictConfig):
        print("TEST DATAMODULE_______________________________________________\n")

        # create a DlibDataModule
        datamodule: DataModule1 = hydra.utils.instantiate(cfg)
        
        # test prepare_data()
        datamodule.prepare_data()

        # test setup()
        datamodule.setup()

        # test batch
        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))
        input = sample_batch[0]
        output = sample_batch[1]
        print(f"Shape of input: {input.shape}")
        print(f"Shape of output: {output.shape}\n")

        # test train_dataloader()
        for batch in tqdm(train_loader):
            pass
        print("train dataloader passed")

        # test val_dataloader()
        val_loader = datamodule.val_dataloader()
        for batch in tqdm(val_loader):
            pass
        print("validation dataloader passed")

        # test test_dataloader()
        test_loader = datamodule.test_dataloader()
        for batch in tqdm(test_loader):
            pass
        print("test dataloader passed")

        print("\nDATAMODULE PASSED_______________________________________________\n")
    
    # set up path
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="dataset1")
    def main(cfg: DictConfig):
        test_datamodule(cfg)
    
    main()
