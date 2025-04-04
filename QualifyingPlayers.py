from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
from CreateFeatures import add_player_game_features
import time

# --- CONFIG ---
ROLLING_WINDOW = 5

# --- QUALIFYING PLAYERS ---
def qualifying_players(team_name):
    """
    Get game log data for each player on the team.
    
    Args:
        team_name (str): The team name.
    
    Returns:
        dict: A dictionary of player IDs and their game log data.
    """
    # Get team ID
    all_teams = teams.get_teams()
    team = next((t for t in all_teams if t['full_name'].lower() == team_name.lower()), None)
    if not team:
        raise ValueError(f"Team name '{team_name}' not found.")

    team_id = team['id']
    roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]

    # Get player IDs
    players = roster['PLAYER'].tolist()
    results = {}

    # Get game log data for each player
    for player in players:
        try:
            df = add_player_game_features(player)
            if df.empty:
                print(f"No game log data for {player}, skipping.")
                continue

            df = df.sort_values(by='GAME_DATE')
            last_5 = df.tail(ROLLING_WINDOW)
            avg_min = last_5['MIN'].mean()
            
            # Skip players with less than 20 minutes
            if avg_min < 20:
                continue

            results[player] = df
        except Exception as e:
            print(f"Failed to train model for {player}: {e}")
        time.sleep(2) # Prevent API rate limiting
    return results
