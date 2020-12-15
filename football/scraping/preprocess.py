import orjson
from itertools import chain
from pathlib import Path

import typer
import joblib
import numpy as np
from tqdm import tqdm
import fastcore.xtras

from ..processing import simplified_wrapper

DATA_FOLDERS = [
    Path("data/episodes/wekick/"),
    # Path("data/episodes/saltyfish/")
]
TARGET_FOLDER = Path("data/steps/")
TARGET_FOLDER.mkdir(parents=True, exist_ok=True)
TARGET_TEAM = "WeKick"


def parse_filepath(filepath: Path, onehot: bool, offense_only: bool = True):
    target_filename = filepath.stem + ".jbl"
    target_filepath = TARGET_FOLDER / target_filename
    if target_filepath.exists():
        print(f"{filepath.stem} exists. Skipped...")
        return
    with open(filepath) as f:
        data = orjson.loads(f.read())
    buffer_features, buffer_targets = [], []
    team_id = int(data["info"]["TeamNames"][1] == TARGET_TEAM)
    if data["info"]["TeamNames"][team_id] != TARGET_TEAM:  # or data["rewards"][team_id] < 0:
        # Skip this one
        print(data["info"]["TeamNames"][team_id], data["rewards"][team_id])
        return
    cnt = 0
    for step in data["steps"]:
        if offense_only:
            # only record offense moves
            owner = step[0]["observation"]["players_raw"][0]["ball_owned_team"]
            if owner == -1:
                continue
            if not step[owner]["action"]:
                continue
            buffer_targets.append(step[owner]["action"][0])
            buffer_features.append(
                simplified_wrapper(step[owner]["observation"]["players_raw"][0])
            )
        else:
            # winner = int(data["rewards"][1] > 0)  # pick the left player in ties
            winner = team_id
            if not step[winner]["action"]:
                continue
            if not step[winner]["observation"]["players_raw"][0]["game_mode"] in (0, 5):
                # only take normal or throwin mode
                cnt += 1
                continue
            features = simplified_wrapper(step[winner]["observation"]["players_raw"][0])
            if (
                features[0] > 0.1 and
                features[2] < 0.3 and
                step[winner]["observation"]["players_raw"][0]["ball_owned_team"] != 0
            ):
                continue
            buffer_targets.append(step[winner]["action"][0])
            buffer_features.append(
                features
            )
    features = np.stack(buffer_features, axis=0)
    targets = np.asarray(buffer_targets)
    joblib.dump([features, targets], target_filepath)
    print(f"Skipped {cnt} steps due to game modes")


def main(onehot: bool = False, offense_only: bool = False):
    files = list(chain(*(x.ls(file_exts=".json") for x in DATA_FOLDERS)))
    for filepath in tqdm(files):
        parse_filepath(filepath, onehot, offense_only=offense_only)


if __name__ == "__main__":
    typer.run(main)
