from torch import nn
import torch

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SimpleUnet(nn.Module):
    def __init__(self, 
                 in_channels: int = 10, 
                 out_channels: int = 1):
        super().__init__()

        # manipulate height & width
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # encoder
        self.conv1 = conv(in_channels, 16)
        self.conv2 = conv(16, 32)

        self.conv3 = conv(32, 64)

        # decoder
        self.up_conv4 = up_conv(64 + 32, 32)
        self.up_conv5 = up_conv(32 + 16, 16)

        # last layer
        self.last_conv = nn.Conv2d(16, out_channels, 1)
    
    def forward(self, x):
        # ENCODER
        conv1 = self.conv1(x)
        # print("Conv1: ", conv1.shape)
        x = self.maxpool(conv1)
        # print("Max pool: ", x.shape, "\n")

        conv2 = self.conv2(x)
        # print("Conv2: ", conv2.shape)
        x = self.maxpool(conv2)
        # print("Max pool: ", x.shape, "\n")

        x = self.conv3(x)
        # print("Conv3: ", x.shape, "\n")
        
        # DECODER
        x = self.upsample(x)
        # print("Up sample: ", x.shape)
        x = torch.cat([x, conv2], dim=1)
        # print("Torch cat: ", x.shape)
        x = self.up_conv4(x)
        # print("Up conv4: ", x.shape, "\n")

        x = self.upsample(x)
        # print("Up sample: ", x.shape)
        x = torch.cat([x, conv1], dim=1)
        # print("Torch cat: ", x.shape)
        x = self.up_conv5(x)
        # print("Up conv5: ", x.shape, "\n")

        # output layer
        return self.last_conv(x)

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
        simple_unet = hydra.utils.instantiate(cfg.net)

        # show model
        summary(model=simple_unet,
                input_size=(16,10, 512, 512),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"])
        
        # test input & output shape
        random_input = torch.randn([16,10, 512, 512])
        output = simple_unet(random_input)
        print(f"\n\nINPUT SHAPE: {random_input.shape}")
        print(f"OUTPUT SHAPE: {output.shape}\n")

    # def main
    @hydra.main(version_base="1.3", config_path=config_path, config_name="dataset3")
    def main(cfg: DictConfig):
        test_module(cfg)

    # call main
    main()