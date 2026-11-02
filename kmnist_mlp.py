"""Mamadou's KMNIST MLP architecture and a seeded validation split."""

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


class KMNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.Flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.linear_relu_stack(self.Flatten(x))


def get_data_loaders(root="./data", batch_size=64, seed=42, download=True):
    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    ])
    full_train = datasets.KMNIST(
        root=root, train=True, transform=transform, download=download,
    )
    test = datasets.KMNIST(
        root=root, train=False, transform=transform, download=download,
    )
    train, val = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        train_loader,
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
