from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv2, leaguedashteamstats, leaguehustlestatsteam, commonteamroster
from nba_api.stats.static import teams
import pandas as pd
from CreateFeatures import add_player_game_features
import numpy as np
import time

ROLLING_WINDOW = 5

# Given a team and the team they play next, create a dataframe for each player on the team with 
# the same features as the model over the most recent games

def create_prediction_df(team_name, opponent_name):
    
    # Get team ID
    all_teams = teams.get_teams()
    team = next((t for t in all_teams if t['full_name'].lower() == team_name.lower()), None)
    if not team:
        raise ValueError(f"Team name '{team_name}' not found.")

    team_id = team['id']
    roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]

    # Get player IDs
    player_ids = roster['PLAYER'].tolist()
    
    # Get game log data for each player
    df = pd.DataFrame()
    for player in player_ids:
        df = pd.concat([df, add_player_game_features(player)], ignore_index=True)
    
    # Filter to only include the last 5 games
    df = df.sort_values(by='GAME_DATE')
    df = df.tail(ROLLING_WINDOW)
    
    # Return the dataframe
    return df

x = create_prediction_df('Golden State Warriors', 'Denver Nuggets')
print(x)