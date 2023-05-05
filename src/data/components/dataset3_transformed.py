from torch.utils.data import Dataset
from .dataset3 import Dataset3
from typing import Optional
import numpy as np
from torchvision import transforms

class TransformedDataset3(Dataset):
    def __init__(self, 
                 dataset: Dataset3, 
                 input_transform: Optional[transforms.Compose] = None,
                 output_transform: Optional[transforms.Compose] = None):
        # set dataset
        self.dataset = dataset

        # set transform
        if input_transform is None or output_transform is None:
            mean = std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            self.input_transform = self.output_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.input_transform = input_transform
            self.output_transform = output_transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        input, output = self.dataset[index]

        input = self.input_transform(input)
        output = self.output_transform(output)

        return input, output

