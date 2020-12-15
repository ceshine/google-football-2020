"""Post-filtering of scraped episodes

Condition(s):

- The minimum updated score of the two teams must be larger than MIN_FINAL_RATING

"""
import orjson
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

from .scraper import list_url

SOURCE_FOLDER = Path("data/episodes/archive")
TARGET_FOLDER = Path("data/episodes/wekick")
TARGET_FOLDER.mkdir(parents=True, exist_ok=True)

MIN_FINAL_RATING = 1300
NUM_TEAMS = 1
# ID of SaltyFish: 5696217 Wekick:5653767
TEAM_ID = 5653767
TEAM_NAME = "WeKick"


def main():
    r = requests.post(list_url, json={"teamId": TEAM_ID})
    rj = r.json()
    teams_df = pd.DataFrame(rj['result']['teams'])
    teams_df.sort_values('publicLeaderboardRank', inplace=True)

    # make df
    team_episodes = pd.DataFrame(rj['result']['episodes'])
    team_episodes['avg_score'] = -1
    for i in range(len(team_episodes)):
        agents = team_episodes['agents'].loc[i]
        team_episodes.loc[i, 'updatedScore'] = min(
            [a['updatedScore'] for a in agents if a['updatedScore'] is not None] or [0]
        )
    team_episodes['final_score'] = team_episodes['updatedScore']
    team_episodes.sort_values('final_score', ascending=False, inplace=True)

    print('{} games for {}'.format(len(team_episodes), teams_df.loc[teams_df.id == TEAM_ID].iloc[0].teamName))

    team_episodes = team_episodes[(team_episodes.final_score > MIN_FINAL_RATING)]
    print('   {} in score range'.format(len(team_episodes)))

    cnt = 0
    for epid in tqdm(team_episodes.id.values):
        source_file = SOURCE_FOLDER / '{}.json'.format(epid)
        if source_file.exists():
            with open(source_file) as f:
                data = orjson.loads(f.read())
            assert TEAM_NAME in data["info"]["TeamNames"], print(data["info"]["TeamNames"])
            source_file.rename(TARGET_FOLDER / '{}.json'.format(epid))
            cnt += 1
    print(f"Moved {cnt} files.")


if __name__ == "__main__":
    main()
