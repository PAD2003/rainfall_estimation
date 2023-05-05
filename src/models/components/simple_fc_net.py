from torch import nn

class SimpleFCNet(nn.Module):
    def __init__(
        self,
        input_size: int = 18,
        hidden_units: int = 8,
        output_size: int = 1,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from torchinfo import summary
    import hydra
    import pyrootutils
    from omegaconf import DictConfig
    import torch

    # ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # set up path
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")

    # func to test DlibModule
    def test_module(cfg: DictConfig) -> None:
        # create a model
        fcn = hydra.utils.instantiate(cfg)

        # show model
        summary(model=fcn,
                input_size=(16,18),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])
        
        # test input & output shape
        fcn = fcn.cpu()
        random_input = torch.randn([16, 18]).cpu()
        output = fcn(random_input)
        print(f"\n\nINPUT SHAPE: {random_input.shape}")
        print(f"OUTPUT SHAPE: {output.shape}\n")

    # def main
    @hydra.main(version_base="1.3", config_path=config_path, config_name="dataset1")
    def main(cfg: DictConfig):
        test_module(cfg)

    # call main
    main()
