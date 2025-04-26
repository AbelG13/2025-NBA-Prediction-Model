import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from nba_api.stats.endpoints import BoxScoreAdvancedV2
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguehustlestatsteam
import time
import numpy as np

# --- MAIN FEATURE FUNCTION ---
def add_player_game_features(name, shift=1):
    """
    Adds rolling and contextual features to a player's game log.
    """

    def playerlog(name):
        playerName = players.find_players_by_full_name(name)[0]
        player_id = playerName['id']
        gamelog = playergamelog.PlayerGameLog(player_id=player_id)
        df = gamelog.get_data_frames()[0]

        df = df.sort_values(by=['Player_ID', 'GAME_DATE'])
        df.to_csv('playerlog.csv', index=False)

    playerlog(name)

    df = pd.read_csv('playerlog.csv', header=0)
    df['Game_ID'] = df['Game_ID'].astype(str)
    df['Player_ID'] = df['Player_ID'].astype(str)


    # Load advanced stats
    df_advanced = pd.read_csv('advanced_stats.csv')
    df_advanced['Player_ID'] = df_advanced['Player_ID'].astype(str)
    df_advanced['Game_ID'] = df_advanced['Game_ID'].astype(str)
    df_advanced.drop(columns=['MIN'], inplace=True)
    # Left join df and df_advanced on Game_ID and Player_ID
    df = pd.merge(df, df_advanced, on=['Game_ID', 'Player_ID'], how='left')
    
    # Create position columns
    positions = ['Center', 'Guard', 'Forward', 'Guard-Forward']

    for pos in positions:
        df[f'Position_{pos}'] = (df['Position'] == pos).astype(int)

    # If Start_Position is NaN set to 0, otherwise set to 1
    df['START_POSITION'] = df['START_POSITION'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Convert date
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')

    # Rolling features
    df['avg_pts_5g'] = df['PTS'].transform(lambda x: x.shift(shift).rolling(5).mean())
    df['avg_min_5g'] = df['MIN'].transform(lambda x: x.shift(shift).rolling(5).mean())
    df['avg_fga_5g'] = df['FGA'].transform(lambda x: x.shift(shift).rolling(5).mean())
    df['avg_fgm_5g'] = df['FGM'].transform(lambda x: x.shift(shift).rolling(5).mean())
    df['avg_fta_5g'] = df['FTA'].transform(lambda x: x.shift(shift).rolling(5).mean())
    df['avg_ftm_5g'] = df['FTM'].transform(lambda x: x.shift(shift).rolling(5).mean())
    df['std_fta_5g'] = df['FTA'].transform(lambda x: x.shift(shift).rolling(5).std())
    df['std_pts_3g'] = df['PTS'].transform(lambda x: x.shift(shift).rolling(3).std())
        
    # Points scored in the previous game
    df['pts_prev_game'] = df['PTS'].shift(shift)

    # Points scored in the previous game per minute
    df['pts_prev_game_per_min'] = df['pts_prev_game'] / df['avg_min_5g']
    
    # Points scored in the previous game per minute squared
    df['pts_prev_game_per_min_squared'] = df['pts_prev_game_per_min'] ** 2

    # Squared terms: avg_fga_5g ** 2
    df['avg_fga_5g_squared'] = df['avg_fga_5g'] ** 2
    # Interaction terms: avg_fga_5g * avg_pts_5g, avg_fga_5g / avg_min_5g
    df['avg_fga_5g_avg_pts_5g'] = df['avg_fga_5g'] * df['avg_pts_5g']
    df['avg_fga_5g_avg_min_5g'] = df['avg_fga_5g'] / df['avg_min_5g']

    # Players season ppg up to current game
    df['ppg'] = df['PTS'].shift(shift).rolling(window=81, min_periods=3).mean()

    # Players season ppg squared
    df['ppg_squared'] = df['ppg'] ** 2

    # Extract Opponent Abbreviation
    df['Opponent'] = df['MATCHUP'].str.split().str[-1]

    # Days since last game
    df['days_since_last_game'] = df['GAME_DATE'].transform(lambda x: x.diff().dt.days)

    # Home or Away
    df['home_or_away'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    df['avg_efg_pct_2g'] = df['EFG_PCT'].transform(lambda x: x.shift(shift).rolling(2).mean())
    df['avg_ts_pct_2g'] = df['TS_PCT'].transform(lambda x: x.shift(shift).rolling(2).mean())
    df['avg_usg_pct_2g'] = df['USG_PCT'].transform(lambda x: x.shift(shift).rolling(2).mean())
    df['avg_off_rating_2g'] = df['OFF_RATING'].transform(lambda x: x.shift(shift).rolling(2).mean())

    # 3-game average
    df['avg_oreb_pct_3g'] = df['OREB_PCT'].transform(lambda x: x.shift(shift).rolling(2).mean())

    # Previous game PIE
    df['pie_prev_game'] = df['PIE'].shift(shift)

    df['weighted_avg_min_3g'] = df['MIN'].shift(shift).rolling(window=3, min_periods=1).apply(
        lambda s: np.average(s, weights=[0.6, 0.3, 0.1][-len(s):])
    )

    # 3. Win/Loss → Previous game outcome (binary)
    df['won_prev_game'] = df['WL'].shift(shift).map({'W': 1, 'L': 0})
    # Feature 4: USG% × MIN (from previous game)
    df['usg_min_product'] = df['USG_PCT'].shift(shift) * df['MIN'].shift(shift)

    # Feature 5: PIE × FGA (from previous game)
    df['pie_fga_product'] = df['PIE'].shift(shift) * df['FGA'].shift(shift)


    # Feature 8: Player OFF_RATING – Opponent DEF_RATING (all from prior game)
    df['off_def_rating_diff'] = df['OFF_RATING'].shift(shift) - df['DEF_RATING']

    # Feature 10: Momentum — was last game PTS above season average (PPG)?
    df['momentum_positive'] = (df['PTS'].shift(shift) > df['ppg']).astype(int)
    
    # Law of Averages Ratio
    df['avg_pts_2g'] = df['PTS'].shift(shift).rolling(window=2).mean()
    df['law_of_averages_ratio'] = np.where(df['ppg'] != 0, df['avg_pts_2g'] / df['ppg'], np.nan)

    # Momentum as difference between last game and game before that
    df['pts_momentum_2g'] = df['PTS'].shift(shift).rolling(window=2).apply(lambda x: x.iloc[-1] - x.iloc[0])
    
    # Rolling Weighted Avg Points (last 3 games)
    df['weighted_avg_pts_3g'] = df['PTS'].rolling(window=3, min_periods=1).apply(
        lambda s: np.average(s, weights=[0.6, 0.3, 0.1][-len(s):])
    )

    return df



def qualifying_players2(team_name):
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
        # try:
            df = add_player_game_features(player)
            if df.empty:
                print(f"No game log data for {player}, skipping.")
                continue

            df = df.sort_values(by='GAME_DATE')
            last_5 = df.tail(5)
            avg_min = last_5['MIN'].mean()
            
            # Skip players with less than 20 minutes
            if avg_min < 20:
                continue

            results[player] = df
        # except Exception as e:
        #     print(f"Failed to train model for {player}: {e}")
            time.sleep(2) # Prevent API rate limiting
    return results

leaguehustlestatsteam = leaguehustlestatsteam.LeagueHustleStatsTeam()
leaguedashteamstats = leaguedashteamstats.LeagueDashTeamStats()

def create_team_df2(roster_dfs : dict[str, pd.DataFrame]):
    """
    Create a team dataframe by merging all player dataframes and joining with opponent team stats.
    
    Args:
        roster_dfs (dict[str, pd.DataFrame]): A dictionary of player dataframes.
    
    Returns:
        pd.DataFrame: A merged dataframe containing player and opponent team stats.
    """

    # Append all dataframes together
    appended_df = pd.concat([df for df in roster_dfs.values()], ignore_index=True)

    # Join on opponent team stats

    df = leaguehustlestatsteam.get_data_frames()[0]
    df2 = leaguedashteamstats.get_data_frames()[0]

    # Create a pandas dataframe with 2 columns: team name and team abbreviation
    dict_team_abbreviation = {}
    all_teams = teams.get_teams()
    for team in all_teams:
        dict_team_abbreviation[team['full_name']] = team['abbreviation']

    name_pd = pd.DataFrame(dict_team_abbreviation.items(), columns=['full_name', 'abbreviation'])

    # Combine league team stats and league hustle stats
    df3 = pd.merge(df, df2, on='TEAM_NAME', how='inner')

    # Combine df3 and name_pd
    df4 = pd.merge(df3, name_pd,  left_on='TEAM_NAME', right_on='full_name', how='left')

    # Drop columns with duplicate names
    df4 = df4.drop(columns=['TEAM_ID_x', 'TEAM_ID_y', 'full_name', 'MIN_y'])
    df5 = df4.rename(columns={'MIN_x': 'MIN'})

    # Prefix every column with "Opp_"
    df6 = df5.add_prefix('OPP_')

    # Join appended_df (on Opponent) and final_df (on abbreviation)
    final_df = pd.merge(appended_df, df6, left_on='Opponent', right_on='OPP_abbreviation', how='left')

    return final_df
