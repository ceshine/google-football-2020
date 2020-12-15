from pathlib import Path

import typer
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import fastcore.xtras

DATA_FOLDER = Path("data/steps/")


def main():
    buffer = []
    for filepath in tqdm(DATA_FOLDER.ls(file_exts=".jbl")):
        features, targets = joblib.load(str(filepath))
        buffer.append(np.concatenate([
            features, targets[:, np.newaxis]
        ], axis=1))
    df = pd.DataFrame(
        np.concatenate(buffer),
        columns=[f"f_{i}" for i in range(features.shape[1])] + ["target"]
    )
    df.to_pickle("data/steps.pd")


if __name__ == "__main__":
    typer.run(main)
