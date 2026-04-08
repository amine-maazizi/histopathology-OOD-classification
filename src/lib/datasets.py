"""Dataset helpers extracted from the notebook."""

from __future__ import annotations

import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode, return_center=False):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        self.return_center = return_center

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())
            if self.return_center:
                self.centers = [int(np.array(hdf.get(img_id).get("metadata"))[0]) for img_id in self.image_ids]
            else:
                self.centers = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, "r") as hdf:
            img = torch.tensor(np.array(hdf.get(img_id).get("img")))
            label = np.array(hdf.get(img_id).get("label")) if self.mode == "train" else None
            center = int(np.array(hdf.get(img_id).get("metadata"))[0]) if self.return_center else None
        if self.return_center:
            return self.preprocessing(img).float(), label, center
        return self.preprocessing(img).float(), label


class CenterBalancedBatchSampler(Sampler):
    """Build batches that contain the same number of samples per center."""

    def __init__(self, centers, batch_size):
        self.centers = list(centers)
        self.batch_size = batch_size

        self.unique_centers = sorted(set(self.centers))
        self.num_centers = len(self.unique_centers)
        if self.num_centers == 0:
            raise ValueError("No centers found for CenterBalancedBatchSampler.")
        if self.num_centers < 2:
            raise ValueError("CenterBalancedBatchSampler requires at least two centers.")
        if self.batch_size % self.num_centers != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by the number of centers ({self.num_centers})."
            )

        self.samples_per_center = self.batch_size // self.num_centers
        self.center_to_indices = {center: [] for center in self.unique_centers}
        for index, center in enumerate(self.centers):
            self.center_to_indices[center].append(index)

        self.num_batches = len(self.centers) // self.batch_size

    def __iter__(self):
        center_pools = {}
        center_offsets = {}
        for center, indices in self.center_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            center_pools[center] = shuffled
            center_offsets[center] = 0

        for _ in range(self.num_batches):
            batch_indices = []
            for center in self.unique_centers:
                offset = center_offsets[center]
                needed = self.samples_per_center
                if offset + needed > len(center_pools[center]):
                    random.shuffle(center_pools[center])
                    offset = 0
                batch_indices.extend(center_pools[center][offset : offset + needed])
                center_offsets[center] = offset + needed

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        super(PrecomputedDataset, self).__init__()
        self.features = features
        self.labels = labels.unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()


def precompute(dataloader, model, device):
    xs, ys = [], []
    for x, y in dataloader:
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)
