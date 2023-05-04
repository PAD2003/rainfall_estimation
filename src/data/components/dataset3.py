from torch.utils.data import Dataset
import os
from typing import List, Tuple
import torch
from torchvision.transforms import ToTensor
from PIL import Image

class Dataset3(Dataset):
    def __init__(self, features_dir, targets_dir) -> None:
        self.features_dir = features_dir
        self.targets_dir = targets_dir

        self.samples: List[Tuple[str, str]] = self.load_data(targets_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        sample: Tuple[str, str] = self.samples[index]

        input = self.image_from_dir(sample[0])
        target = self.image_from_dir(sample[1])

        return input, target
    
    def load_data(self, radar_dir) -> List[Tuple[str, str]]:
        """Set features and targets"""
        result = []
        
        for root, dirs, files in os.walk(radar_dir):
            for file in files:
                if file.endswith(".tif"):
                    path = os.path.join(root, file)
                    folder_path = os.path.dirname(path)
                    folder_path = "/".join(folder_path.split("/")[-4:])
                    # result.append((f"data/Dataset3/himawari/{folder_path}", f"data/Dataset3/radar/{folder_path}"))
                    result.append((self.features_dir + "/" + folder_path, self.targets_dir + "/" + folder_path))
                    # /mnt/banana/k66/pad/rainfall_estimation/data//Dataset3/himawari/k66/pad/rainfall_estimation/data//Dataset3/radar/2020/10/18/12
        
        return result
    
    def image_from_dir(self, feature_dir)->torch.Tensor:
        """Stack all image in a directory"""
        res = []

        for path in os.listdir(feature_dir):
            image_path = os.path.join(feature_dir, path)
            image = Image.open(image_path)
            tensor_image = ToTensor()(image).squeeze()
            res.append(tensor_image)

        return torch.stack(res)