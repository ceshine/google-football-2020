import random

import torch
import joblib
import numpy as np
from kaggle_environments.envs.football.helpers import *
from football.processing import simplified_wrapper
from football.models import MlpClassifierModel, MoeClassifierModel

# load the model from disk
# model = MoeClassifierModel(164)
model = MlpClassifierModel(153)
filename = 'model.pth'
try:
    model.load_state_dict(torch.load(filename, map_location="cpu"))
except FileNotFoundError:
    model.load_state_dict(torch.load('/kaggle_simulations/agent/' + filename, map_location="cpu"))
model.eval()

try:
    scaler = joblib.load("scaler.jbl")
except FileNotFoundError:
    scaler = joblib.load('/kaggle_simulations/agent/scaler.jbl')

directions = [[Action.TopLeft, Action.Top, Action.TopRight],
              [Action.Left, Action.Idle, Action.Right],
              [Action.BottomLeft, Action.Bottom, Action.BottomRight]]

PERFECT_RANGE = [[0.9, 1.], [-0.06, 0.06]]
DANGER_RANGE_1 = [[.98, 1.2], [0.15, .3]]
DANGER_RANGE_2 = [[.98, 1.2], [-.3, -.15]]
DANGER_RANGE_3 = [[.98, 1.2], [.3, .45]]
DANGER_RANGE_4 = [[.98, 1.2], [-45., -.3]]
DANGER_RANGE_5 = [[-1., 1.], [.4, .45]]
DANGER_RANGE_6 = [[-1., 1.], [-.45, -.4]]
DEFEND_RANGE = [[-1.2, -0.8], [-.5, .5]]
GOOD_RANGE = [[0.7, 0.95], [-0.1, 0.1]]


def get_distance(pos1, pos2):
    return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5


def inside(pos, area):
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]


def dirsign(x):
    return 1 if abs(x) < 0.02 else (0 if x < 0 else 2)


def get_action(action_num):
    if action_num == 0:
        return Action.Idle
    if action_num == 1:
        return Action.Left
    if action_num == 2:
        return Action.TopLeft
    if action_num == 3:
        return Action.Top
    if action_num == 4:
        return Action.TopRight
    if action_num == 5:
        return Action.Right
    if action_num == 6:
        return Action.BottomRight
    if action_num == 7:
        return Action.Bottom
    if action_num == 8:
        return Action.BottomLeft
    if action_num == 9:
        return Action.LongPass
    if action_num == 10:
        return Action.HighPass
    if action_num == 11:
        return Action.ShortPass
    if action_num == 12:
        return Action.Shot
    if action_num == 13:
        # return Action.Sprint
        return Action.ReleaseDirection
    # if action_num == 14:
    #     return Action.ReleaseDirection
    #     # return Action.Right
    # if action_num == 15:
    #     return Action.ReleaseSprint
    # if action_num == 16:
    #     # return Action.Sliding
    #     return Action.Idle
    # if action_num == 17:
    #     # return Action.Dribble
    #     return Action.Idle
    # if action_num == 18:
    #     # return Action.ReleaseDribble
    #     return Action.Idle
    return Action.Right


def agent(obs):
    obs = obs['players_raw'][0]
    # print(", ".join([f"{x:.4f}" for x in features[0]]))

    controlled_player_pos = obs['left_team'][obs['active']]
    if obs["game_mode"] in (GameMode.Penalty.value, GameMode.GoalKick.value):
        return [Action.Shot.value]
    if obs["game_mode"] == GameMode.KickOff.value:
        return [Action.Right.value]
        # if random.random() < 0.5:
        #     return [Action.ShortPass.value]
        # return [Action.HighPass.value]
    if obs["game_mode"] == GameMode.Corner.value:
        if random.random() < 0.5:
            return [Action.Shot.value]
        if random.random() < 0.4:
            return [Action.LongPass.value]
        return [Action.HighPass.value]
    if obs["game_mode"] == GameMode.FreeKick.value:
        return [Action.Right.value]
        # if random.random() < 0.6:
        #     return [Action.Shot.value]
        # if random.random() < 0.6:
        #     return [Action.HighPass.value]
        # return [Action.ShortPass.value]

    ball_targetx = obs['ball'][0]+(obs['ball_direction'][0]*6)
    ball_targety = obs['ball'][1]+(obs['ball_direction'][1]*6)
    ball_targetz = obs['ball'][2]+(obs['ball_direction'][2]*6)
    ball_projected = [ball_targetx, ball_targety]

    if obs['ball_owned_team'] in (-1, 1):
        e_dist = get_distance(obs['left_team'][obs['active']], obs['ball'])
        if e_dist > .08 and ball_targetz < 0.3:
            if -1 < controlled_player_pos[0] < 0.6 and not obs['sticky_actions'][8]:
                print("Enforce Sprint")
                return [Action.Sprint.value]
            elif controlled_player_pos[0] < 0.9 and not obs['sticky_actions'][8] and obs['ball_owned_team'] == 1:
                print("Enforce Sprint")
                return [Action.Sprint.value]
            # if not obs['sticky_actions'][8]:
            #     return [Action.Sprint.value]
            # Run where ball will be
            xdir = dirsign(ball_targetx - controlled_player_pos[0])
            ydir = dirsign(ball_targety - controlled_player_pos[1])
            # print("Chasing ball", directions[ydir][xdir], "%.3f %.2f %.2f" % (e_dist, xdir, ydir))
            return [directions[ydir][xdir].value]
        if (
            e_dist < .03 and ball_targetz < 0.2 and
            controlled_player_pos[0] < obs['ball'][0] and
            obs['ball_owned_team'] != 0 and
            obs['left_team'][obs['active']][0] < 0 and
            obs['left_team_direction'][obs['active']][0] > 0
        ):
            if obs['sticky_actions'][8]:
                print("Enforce Defensive ReleaseSprint")
                return [Action.ReleaseSprint.value]

    # Make sure player is running.
    if -1 < ball_projected[0] < 0.75 and obs['ball_owned_team'] == 0 and not obs['sticky_actions'][8]:
        print("Enforce Sprint")
        return [Action.Sprint.value]
    elif (
        0.8 < ball_projected[0] and obs['ball_owned_team'] == 0 and obs['sticky_actions'][8]
        and obs['left_team_direction'][obs['active']][0] > 0
    ):
        print("Enforce ReleaseSprint")
        return [Action.ReleaseSprint.value]

    # if (
    #     obs['ball_owned_team'] == 0
    #     and inside(controlled_player_pos, PERFECT_RANGE)
    #     and obs['left_team_direction'][obs['active']][0] >= 0
    # ):
    #     print("Enforce Shot")
    #     return [Action.Shot.value]

    if (
        obs['ball_owned_team'] == 0
        and inside(ball_projected, GOOD_RANGE)
        and (
            np.max(
                np.asarray(obs['right_team'])[:, 0]
            ) < obs['left_team'][obs['active']][0]
        )
        and (
            obs['ball_direction'][1] * obs['ball'][1] > 0
        )
    ):
        print("***Enforce Shot***")
        return [Action.Shot.value]

    # if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and (inside(ball_projected, DANGER_RANGE_1) or inside(ball_projected, DANGER_RANGE_2)):
    #     print("Enforce Pass")
    #     if random.random() < 0.75:
    #         return [Action.LongPass.value]
    #     return [Action.HighPass.value]

    if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and (inside(ball_projected, DANGER_RANGE_1) or inside(ball_projected, DANGER_RANGE_3)):
        print("...Enforce Goal Line turn...")
        if random.random() < 0.5:
            return [Action.Top.value]
        return [Action.TopLeft.value]

    if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and (inside(ball_projected, DANGER_RANGE_2) or inside(ball_projected, DANGER_RANGE_4)):
        print("...Enforce Goal Line turn...")
        if random.random() < 0.5:
            return [Action.Bottom.value]
        return [Action.BottomLeft.value]

    # if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and (inside(ball_projected, DANGER_RANGE_3) or inside(ball_projected, DANGER_RANGE_4)):
    #     print("Enforce Pass")
    #     if random.random() < 0.75:
    #         return [Action.ShortPass.value]
    #     return [Action.LongPass.value]

    # if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and (inside(ball_projected, DANGER_RANGE_5) or inside(ball_projected, DANGER_RANGE_6)):
    #     print("++Enforce Edge Pass++")
    #     if random.random() < 0.75:
    #         return [Action.ShortPass.value]
    #     return [Action.LongPass.value]

    if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and inside(ball_projected, DANGER_RANGE_5):
        print("++Enforce Edge Turn++")
        if random.random() < 0.5:
            return [Action.TopRight.value]
        return [Action.Top.value]

    if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and inside(ball_projected, DANGER_RANGE_6):
        print("++Enforce Edge Turn++")
        if random.random() < 0.5:
            return [Action.BottomRight.value]
        return [Action.Bottom.value]

    # if (obs['ball_owned_team'] == 0 and obs["game_mode"] == GameMode.Normal.value) and inside(ball_projected, DEFEND_RANGE):
    #     print("***Emergency Defense***")
    #     # if obs["active"] == 0:
    #     #     return [Action.LongPass.value]
    #     if obs['left_team_direction'][obs['active']][0] <= 0 or random.random() < 0.2:
    #         return [random.choice([
    #                 Action.BottomRight,
    #                 Action.TopRight]).value]
    #     if random.random() < 0.4:
    #         return [Action.HighPass.value]
    #     return [Action.Shot.value]

    features = simplified_wrapper(obs)[np.newaxis, :]
    features = scaler.transform(features)
    # features = reference_wrapper(obs)[np.newaxis, :]
    input_tensor = torch.tensor(features, dtype=torch.float)
    logits = model(input_tensor)[0, :]
    if obs['ball_owned_team'] == 0 and random.random() < 0.8:
        dist = torch.distributions.Categorical(logits=logits * 2)
        predicted = dist.sample()
    else:
        predicted = torch.argmax(logits)
    do = get_action(predicted)
    # if do == Action.Shot and obs['left_team'][obs['active']][0] < 0:
    #     # Don't use shot in our side of the court
    #     print("///Change Shot to High Pass///")
    #     return [Action.HighPass.value]
    if do in (Action.HighPass, Action.LongPass, Action.ShortPass, Action.Shot, Action.Sprint, Action.ReleaseSprint, Action.ReleaseDirection):
        print(do, do.value)
    return [do.value]
