import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# --- CONFIGURABLE PARAMS ---
ROLLING_WINDOW = 5

# --- MAIN FEATURE FUNCTION ---
def add_player_game_features(name):
    """
    Adds rolling and contextual features to a player's game log.
    Game_log_df has columns: ['GAME_DATE', 'PLAYER_ID', 'TEAM_ID', 'OPPONENT_ID', 'PTS', 'AST', 'MIN', 'FGA', 'FTA', 'TOV', 'MATCHUP']
    """

    playerName = players.find_players_by_full_name(name)[0]
    player_id = playerName['id']
    gamelog = playergamelog.PlayerGameLog(player_id=player_id)
    df = gamelog.get_data_frames()[0]

    df = df.sort_values(by=['Player_ID', 'GAME_DATE'])

    # Convert date
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')

    # Rolling features
    df['avg_pts_5g'] = df['PTS'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean())
    df['avg_min_5g'] = df['MIN'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean())
    df['avg_fga_5g'] = df['FGA'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean())
    df['avg_fgm_5g'] = df['FGM'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean())
    df['avg_fta_5g'] = df['FTA'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean())
    df['avg_ftm_5g'] = df['FTM'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).mean())
    df['std_fta_5g'] = df['FTA'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW).std())
    # Squared terms: avg_fga_5g ** 2
    df['avg_fga_5g_squared'] = df['avg_fga_5g'] ** 2
    # Interaction terms: avg_fga_5g * avg_pts_5g, avg_fga_5g / avg_min_5g
    df['avg_fga_5g_avg_pts_5g'] = df['avg_fga_5g'] * df['avg_pts_5g']
    df['avg_fga_5g_avg_min_5g'] = df['avg_fga_5g'] / df['avg_min_5g']


    # OPPONENT FEATURES
    # Extract Opponent Abbreviation
    df['Opponent'] = df['MATCHUP'].str.split().str[-1]

    # Days since last game
    df['days_since_last_game'] = df['GAME_DATE'].transform(lambda x: x.diff().dt.days)

    # Home or Away
    df['home_or_away'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    return df

