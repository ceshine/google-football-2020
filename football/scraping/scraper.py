"""
https://www.kaggle.com/felipebihaiek/google-football-episode-scraper-quick-fix
"""
import os
from pathlib import Path

import requests
import pandas as pd

DATA_FOLDER = Path("data/episodes/archive/")
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

MIN_FINAL_RATING = 1350
NUM_TEAMS = 1
EPISODES = 100

BUFFER = 1
# ID of SaltyFish: 5696217 Wekick:5653767
TEAM_ID = 5653767
base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
list_url = base_url + "ListEpisodes"


def saveEpisode(epid):
    # request
    re = requests.post(get_url, json={"EpisodeId": int(epid)})

    # save replay
    with open(DATA_FOLDER / '{}.json'.format(epid), 'w') as f:
        f.write(re.json()['result']['replay'])


def main():
    all_files = []
    for root, dirs, files in os.walk(DATA_FOLDER / "..", topdown=False):
        all_files.extend(files)
    seen_episodes = [
        int(f.split('.')[0]) for f in all_files
        if '.' in f and f.split('.')[0].isdigit() and f.split('.')[1] == 'json'
    ]
    print('{} games in existing library'.format(len(seen_episodes)))

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
            [a['updatedScore']
                for a in agents if a['updatedScore'] is not None] or [0]
        )
    team_episodes['final_score'] = team_episodes['updatedScore']
    team_episodes.sort_values('final_score', ascending=False, inplace=True)

    print('{} games for {}'.format(len(team_episodes),
                                   teams_df.loc[teams_df.id == TEAM_ID].iloc[0].teamName))

    team_episodes = team_episodes[(
        team_episodes.final_score > MIN_FINAL_RATING)]
    print('   {} in score range'.format(len(team_episodes)))

    team_df = team_episodes[~team_episodes.id.isin(seen_episodes)]
    print('      {} remain to be downloaded\n'.format(len(team_df)))

    i = 0
    while i < len(team_df) and i < EPISODES:
        epid = team_df.id.iloc[i]
        saveEpisode(epid)
        size = os.path.getsize(DATA_FOLDER / '{}.json'.format(epid)) / 1e6
        print('{}: Saved Episode #{} @ {:.1f}MB'.format(i, epid, size))
        i += 1


if __name__ == "__main__":
    main()
