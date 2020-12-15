import enum
import math

import numpy as np


def onehot_feature(num, num_max):
    arr = np.zeros(num_max)
    arr[num] = 1
    return arr


def sort_and_stack(arr1, *arr):
    """sort by arr1"""
    sorted_idx = np.argsort(arr1)
    return np.stack([arr1[sorted_idx]] + [x[sorted_idx] for x in arr], axis=1)


def get_stats(team_pos, team_dir):
    team_distance = np.sqrt((team_pos ** 2).sum(axis=1))
    team_orient = np.arctan2(team_pos[:, 1], team_pos[:, 0])
    team_orient[team_orient < 0] = team_orient[team_orient < 0] + 2 * math.pi
    team_pos_projected = team_pos + team_dir * 3
    # team_distance_projected = np.sqrt((team_pos_projected ** 2).sum(axis=1))
    # team_orient_projected = np.arctan2(team_pos_projected[:, 1], team_pos_projected[:, 0])
    # team_orient_projected[team_orient_projected < 0] = team_orient_projected[team_orient_projected < 0] + 2 * math.pi
    return sort_and_stack(
        team_distance, team_orient,
        # team_distance_projected, team_orient_projected,
        team_pos[:, 0], team_pos[:, 1],
        team_pos_projected[:, 0], team_pos_projected[:, 1]
        # team_dir[:, 0], team_dir[:, 1]
    )
    # return np.stack([team_distance, team_orient], axis=1)


def extract_enum_value(obj):
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


def extract_features(obs, onehot: bool = False):
    # -1 = ball not owned, 0 = left team, 1 = right team.
    owner = extract_enum_value(obs["ball_owned_team"]) + 1
    if onehot:
        owner = onehot_feature(owner, 3)
    active_player = obs["active"]
    role = extract_enum_value(obs["left_team_roles"][active_player])
    if onehot:
        role = onehot_feature(role, 10)
    left_team_pos = np.array(obs["left_team"])
    left_team_dir = np.array(obs["left_team_direction"])
    active_player_pos = left_team_pos[active_player].copy()
    active_player_dir = left_team_dir[active_player].copy()
    left_team_pos -= active_player_pos
    assert obs["left_team_roles"][0] == 0
    left_goalkeeper_pos = left_team_pos[0]
    left_team_stats = get_stats(left_team_pos[1:], left_team_dir[1:])[1:]
    right_team_pos = np.array(obs["right_team"]) - active_player
    right_team_dir = np.array(obs["right_team_direction"])
    assert obs["right_team_roles"][0] == 0
    right_goalkeeper_pos = right_team_pos[0]
    right_team_stats = get_stats(right_team_pos[1:], right_team_dir[1:])
    goalkeeper_stats = get_stats(
        np.stack([left_goalkeeper_pos, right_goalkeeper_pos]),
        np.stack([left_team_dir[0], right_team_dir[0]])
    )
    game_mode = extract_enum_value(obs["game_mode"])
    ball_pos = np.array(obs["ball"])
    ball_relative_pos = ball_pos[:2] - active_player_pos
    # Offside detection
    max_right_team_x = np.max(np.array(obs["right_team"])[1:, 0])
    max_left_team_x = np.max(
        np.array(obs["left_team"])[
            np.array([x != active_player for x in range(len(obs["left_team"]))]), 0
        ]
    )
    # Debug
    # if max_right_team_x < max_left_team_x:
    #     print(max_right_team_x, max_left_team_x)
    potential_offside = max_left_team_x - max_right_team_x
    if onehot:
        game_mode = onehot_feature(game_mode, 7)
    return {
        "owner": owner if onehot else [owner],
        "role": role if onehot else [role],
        "role_simplified": np.array([
            obs["left_team_roles"][active_player] == 0,  # goalkeeper
            obs["left_team_roles"][active_player] in (1, 2, 3, 4),  # back and defense midfield
            obs["left_team_roles"][active_player] in (5, 6, 7),  # midfield
            obs["left_team_roles"][active_player] in (8, 9),  # attack midfield and front
        ]),
        "left_team": left_team_stats,
        "right_team": right_team_stats,
        "goalkeeper_stats": goalkeeper_stats,
        "active_player_pos": active_player_pos,
        "active_player_dir": active_player_dir,
        "active_player_pos_expected": active_player_pos + active_player_dir * 3,
        "game_mode": game_mode if onehot else [game_mode],
        "game_mode_simplified": np.array([
            obs["game_mode"] == 0,  # normal
            obs["game_mode"] == 5,  # throwIn
            # game_mode in (1, 2, 3, 4, 6)  # others
        ]),
        "ball_stats": np.concatenate([
            np.array([
                get_distance(ball_pos, np.array(obs["left_team"])[active_player]),
                get_heading(ball_pos, np.array(obs["left_team"])[active_player])
            ]),
            ball_pos[:2],
            [min(ball_pos[2], 1.75)],
            [min(ball_pos[2] + obs["ball_direction"][2] * 3, 2.)],
            ball_pos[:2] + np.array(obs["ball_direction"])[:2] * 3,
            ball_relative_pos,
            ball_relative_pos + np.array(obs["ball_direction"])[:2] * 3
        ]),
        "sticky_actions": np.array(obs["sticky_actions"]),
        "potential_offside": [potential_offside]
    }


def simplified_wrapper(obs):
    features = extract_features(obs, onehot=True)
    arr = np.concatenate([
        features["ball_stats"],
        features["owner"],
        features["role_simplified"],
        features["game_mode_simplified"],
        features["active_player_pos"],
        features["active_player_pos_expected"],
        features["goalkeeper_stats"].reshape(-1),
        features["right_team"].reshape(-1),
        features["left_team"].reshape(-1),
        features["sticky_actions"][8:9],  # [np.array([8, 9])],
        features["potential_offside"]
    ])
    return arr


def basic_wrapper(obs, onehot: bool = False):
    features = extract_features(obs, onehot=onehot)
    arr = np.concatenate([
        features["ball_stats"],
        features["owner"],
        features["role"], features["game_mode"],
        features["active_player_pos"],
        features["active_player_pos_expected"],
        features["goalkeeper_stats"].reshape(-1),
        # features["right_team"][:5, :].reshape(-1),
        features["right_team"].reshape(-1),
        features["left_team"].reshape(-1),
        features["sticky_actions"][8:9],  # [np.array([8, 9])],
        features["potential_offside"]
    ])
    # print(features["sticky_actions"][:8])
    return arr


def get_distance(pos1, pos2):
    return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5


def get_heading(pos1, pos2):
    raw_head = math.atan2(pos1[0]-pos2[0], pos1[1]-pos2[1])/math.pi*180

    if raw_head < 0:
        head = 360+raw_head
    else:
        head = raw_head
    return head


def reference_wrapper(obs):
    controlled_player_pos = obs['left_team'][obs['active']]
    x = controlled_player_pos[0]
    y = controlled_player_pos[1]

    to_append = []
    goalx = 0.0
    goaly = 0.0

    sidelinex = 0.0
    sideliney = 0.42

    goal_dist = get_distance((x, y), (goalx, goaly))
    sideline_dist = get_distance((x, y), (sidelinex, sideliney))
    to_append.append(goal_dist)
    to_append.append(sideline_dist)

    for i in range(len(obs['left_team'])):
        dist = get_distance((x, y), (obs['left_team'][i][0], obs['left_team'][i][1]))
        head = get_heading((x, y), (obs['left_team'][i][0], obs['left_team'][i][1]))
        to_append.append(dist)
        to_append.append(head)

    for i in range(len(obs['right_team'])):
        dist = get_distance((x, y), (obs['right_team'][i][0], obs['right_team'][i][1]))
        head = get_heading((x, y), (obs['right_team'][i][0], obs['right_team'][i][1]))
        to_append.append(dist)
        to_append.append(head)

    return np.array(to_append)
