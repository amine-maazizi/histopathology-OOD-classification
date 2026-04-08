"""Inference helpers extracted from the notebook."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm


def load_test_ids(test_images_path):
    """Load the test image IDs from the HDF5 file."""

    with h5py.File(test_images_path, "r") as hdf:
        return list(hdf.keys())

@torch.no_grad()
def predict_test(test_ids, preprocessing, feature_extractor, linear_probing, device, test_images_path="test.h5"):
    """Run the notebook's test-time prediction loop."""

    predictions = []
    with h5py.File(test_images_path, "r") as hdf:
        for test_id in tqdm(test_ids, leave=False):
            img = preprocessing(torch.tensor(np.array(hdf.get(test_id).get("img")))).unsqueeze(0).float()
            pred = linear_probing(feature_extractor(img.to(device))).detach().cpu()
            predictions.append(int(pred.item() > 0.5))
    return predictions


@torch.no_grad()
def predict_test_tta(test_ids, eval_preprocessing, tta_augmentations, feature_extractor, linear_probing, device, test_images_path="test.h5"):
    """Run inference with test-time augmentation. Averages predictions over the original image and all augmented views."""

    predictions = []
    with h5py.File(test_images_path, "r") as hdf:
        for test_id in tqdm(test_ids, leave=False):
            raw_img = torch.tensor(np.array(hdf.get(test_id).get("img")))
            views = [eval_preprocessing(raw_img).float()] + [aug(raw_img).float() for aug in tta_augmentations]
            batch = torch.stack(views).to(device)
            probs = linear_probing(feature_extractor(batch)).cpu()
            predictions.append(int(probs.mean().item() > 0.5))
    return predictions


def write_submission(test_ids, predictions, output_csv="baseline.csv"):
    """Write the submission CSV with the expected ID and Pred columns."""

    solutions_data = pd.DataFrame({"ID": [int(test_id) for test_id in test_ids], "Pred": predictions})
    output_csv = Path(output_csv)
    solutions_data.to_csv(output_csv, index=False)
    return output_csv
