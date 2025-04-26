import pandas as pd
import time
from PredictionModel import process_and_model_all_teams
from CreateFeatures import add_player_game_features, create_team_df2, qualifying_players2
def run(recompile=False):
    if recompile:
        all_teams = [
            "Boston Celtics",
            "Cleveland Cavaliers", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
            "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
            "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New York Knicks",
            "Oklahoma City Thunder", "Orlando Magic"
        ]

        # Create team df for every team and append them together then save as a csv
        df = pd.DataFrame()
        count = 0
        for team in all_teams:
            df = pd.concat([df, create_team_df2(qualifying_players2(team))], ignore_index=True)
            count += 1
            print(f"Processed {team}")
            if count % 3 == 0:
                time.sleep(2)

        df.to_csv('new_all_teams.csv', index=False)

    # Last minute feature additions
    df = pd.read_csv('new_all_teams.csv')

    df['fga_vs_opp_def'] = df['avg_fga_5g'] * (1 - df['OPP_FG_PCT_RANK'] / 30)

    df.to_csv('new_all_teams.csv', index=False)

    process_and_model_all_teams('new_all_teams.csv')

run(recompile=True)

# # Get lebron game log with nba_api
# from nba_api.stats.endpoints import playergamelog

# lebron = playergamelog.PlayerGameLog(player_id=202695)

# df = lebron.get_data_frames()[0]

# df.sort_values(by="GAME_DATE", ascending=False, inplace=True)
# print(df.head(20))
