import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import joblib
import typer
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_helper_bot import (
    BaseBot, LearningRateSchedulerCallback,
    Top1Accuracy,
    MovingAverageStatsTrackerCallback,
    CheckpointCallback,
    MultiStageScheduler, LinearLR
)
from pytorch_helper_bot.optimizers import RAdam

from .models import MlpClassifierModel, MoeClassifierModel

INPUT_FILE_PATH = Path("data/steps.pd")
CACHE_DIR = Path("cache/")


def get_dataloader(batch_size: int = 32, test_size: float = 0.05):
    df = pd.read_pickle(INPUT_FILE_PATH)
    df = df[(df.target <= 12) | (df.target == 14)]
    df.loc[df.target == 14, "target"] = 13
    X = df.drop(['target'], axis=1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    df.target = df.target.astype(int)
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.long)
    )
    valid_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.long)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, num_workers=1, pin_memory=True, drop_last=True, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size*2, num_workers=1, pin_memory=True, drop_last=False),
        X.shape[1],
        scaler
    )


def main(lr: float = 2e-3, epochs: int = 10, batch_size: int = 128):
    train_dl, valid_dl, n_features, scaler = get_dataloader(batch_size=batch_size, test_size=0.05)
    joblib.dump(scaler, CACHE_DIR / "scaler.jbl")
    print(f"# of Features: {n_features}")
    model = MlpClassifierModel(n_features).cuda()
    # model = MoeClassifierModel(n_features).cuda()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-2)
    optimizer = RAdam(model.parameters(), lr=lr)
    total_steps = len(train_dl) * epochs

    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir=CACHE_DIR / "model_cache/",
        monitor_metric="accuracy"
    )
    lr_durations = [
        int(total_steps*0.1),
        int(np.ceil(total_steps*0.9))
    ]
    break_points = [0] + list(np.cumsum(lr_durations))[:-1]
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=len(train_dl),
            log_interval=len(train_dl) // 2
        ),
        LearningRateSchedulerCallback(
            MultiStageScheduler(
                [
                    LinearLR(optimizer, 0.01, lr_durations[0]),
                    CosineAnnealingLR(optimizer, lr_durations[1], 1e-6)
                ],
                start_at_epochs=break_points
            )
        ),
        checkpoints,
    ]
    class_weights = torch.ones(14).cuda()
    class_weights[12] = 1.25  # Shot
    bot = BaseBot(
        log_dir=CACHE_DIR / "logs",
        model=model, train_loader=train_dl,
        valid_loader=valid_dl, clip_grad=10.,
        optimizer=optimizer, echo=True,
        criterion=nn.CrossEntropyLoss(weight=class_weights),
        callbacks=callbacks,
        metrics=(Top1Accuracy(),),
        pbar=False, use_tensorboard=False,
        use_amp=False
    )
    bot.train(
        total_steps=total_steps,
        checkpoint_interval=len(train_dl)
    )
    bot.load_model(checkpoints.best_performers[0][1])
    torch.save(bot.model.state_dict(), CACHE_DIR /
               "final_weights.pth")
    print("Model saved")
    checkpoints.remove_checkpoints(keep=0)
    print("Checkpoint removed")


if __name__ == "__main__":
    typer.run(main)
