# MVA DLMI 2026 - Histopathology OOD Classification

## Overview
This student project targets histopathology out-of-distribution classification for the MVA DLMI 2026 challenge.
The task is to classify image patches as tumor vs non-tumor.
The main difficulty is distribution shift across medical centers.
This repository provides a baseline pipeline (DINOv2 + linear probing) and a small modular codebase for experiments.

## Project Structure
- getting_started.ipynb: main notebook to run experiments end to end.
- src/: modular code for datasets, models, training, inference, and solution definitions.
- train.h5, val.h5, test.h5: challenge datasets.
- baseline.csv, best_model.pth: typical prediction and checkpoint outputs.

## How to Run
1. Install dependencies:
   - pip install -r requirements.txt
   - Main packages: PyTorch, torchvision, torchmetrics
2. Open getting_started.ipynb.
3. Run notebook cells, or use the solution API:

```python
from src.main import get_solution

config = {
    "train_path": "train.h5",
    "val_path": "val.h5",
    "test_path": "test.h5",
    "batch_size": 16,
}

solution = get_solution("baseline", config)
solution.fit()
solution.predict_test("baseline.csv")
```
