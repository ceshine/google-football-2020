import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import joblib
import typer
import xgboost as xgb
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from pytorch_helper_bot.optimizers import RAdam

from .models import MlpClassifierModel, MoeClassifierModel

INPUT_FILE_PATH = Path("data/steps.pd")
CACHE_DIR = Path("cache/")


def get_data(test_size: float = 0.05):
    df = pd.read_pickle(INPUT_FILE_PATH)
    df = df[(df.target <= 12) | (df.target == 14)]
    df.loc[df.target == 14, "target"] = 13
    X = df.drop(['target'], axis=1).values
    df.target = df.target.astype(int)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    return dtrain, dvalid, X.shape[1]


def main(num_round: int = 100, seed: int = 42):
    dtrain, dvalid, n_features = get_data(test_size=0.05)
    print(f"# of Features: {n_features}")
    params = {
        'nthread': 4,
        'tree_method': 'gpu_hist',
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'max_depth': 7,
        'eta': 0.2,
        'gamma': 0.02,
        'num_class': 14,
        'objective': 'multi:softprob',
        'eval_metric': ['mlogloss', 'merror'],
        'seed': seed
    }
    eval_list = [(dvalid, 'eval'), (dtrain, 'train')]

    bst = xgb.train(params, dtrain, num_round, eval_list, early_stopping_rounds=20)
    bst.save_model(CACHE_DIR / 'xgb.model')


if __name__ == "__main__":
    typer.run(main)
