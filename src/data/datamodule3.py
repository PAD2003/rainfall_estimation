from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from components.dataset3 import Dataset3
from components.dataset3_transformed import TransformedDataset3
from lightning import LightningDataModule
from torchvision import transforms

class DataModule3(LightningDataModule):
    def __init__(self,
                 dataset: Dataset3,
                 data_dir: str,
                 input_transform: transforms = None,
                 output_transform: transforms = None,
                 train_val_test_split = [5666, 1000, 0],
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False) -> None:
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed"""
        pass

    def setup(self, stage: Optional[str]=None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            data_train, data_val, data_test = random_split(
                dataset=self.hparams.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_train = TransformedDataset3(dataset=data_train, input_transform=self.hparams.input_transform, output_transform=self.hparams.output_transform)
            self.data_val = TransformedDataset3(dataset=data_val, input_transform=self.hparams.input_transform, output_transform=self.hparams.output_transform)
            self.data_test = TransformedDataset3(dataset=data_val, input_transform=self.hparams.input_transform, output_transform=self.hparams.output_transform)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data_test,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from tqdm import tqdm

    # ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # set up path
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")

    # test dataset
    def test_dataset(cfg: DictConfig):
        print("TEST DATASET\n")

        # create a dataset
        dataset: Dataset3 = hydra.utils.instantiate(cfg.dataset)

        # test __len__()
        print(f"Length of train dataset: {len(dataset)}\n")

        # test __getitem__()
        features, targets = dataset[0]

        # test a sample
        print(f"Shape of feature: {features.shape}")
        print(f"Shape of targets: {targets.shape}")

        print("\nDATASET PASSED\n")

    # test datamodule
    def test_datamodule(cfg: DictConfig):
        print("TEST DATAMODULE\n")

        # create a DlibDataModule
        datamodule: Dataset3 = hydra.utils.instantiate(cfg)
        
        # test prepare_data()
        datamodule.prepare_data()

        # test setup()
        datamodule.setup()

        # test batch
        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))
        features = sample_batch[0]
        targets = sample_batch[1]
        print(f"Shape of features: {features.shape}")
        print(f"Shape of targets: {targets.shape}\n")

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

        print("\nDATAMODULE PASSED\n")

    # def main with hydra
    @hydra.main(version_base="1.3", config_path=config_path, config_name="test_dataset3")
    def main(cfg: DictConfig):
        test_dataset(cfg)
        test_datamodule(cfg)
        # print(OmegaConf.to_yaml(cfg))

    # call main
    main()