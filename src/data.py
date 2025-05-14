import numpy as np

import torch.utils
from torchvision import transforms
from torch.utils.data import DataLoader

from load_dataset import ImageDataset
from sklearn.model_selection import StratifiedShuffleSplit

image_transforms = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225])
    ])

def data_load(data_path: str, batch_size: int):

    dataset = ImageDataset(toor_dir=data_path, transform=image_transforms)

    # Stratified Sampling
    labels = np.array(dataset.labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_indices, test_indices in splitter.split(np.zeros(len(labels)), labels):
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

