from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguehustlestatsteam
from nba_api.stats.static import teams
import pandas as pd

leaguehustlestatsteam = leaguehustlestatsteam.LeagueHustleStatsTeam()
leaguedashteamstats = leaguedashteamstats.LeagueDashTeamStats()

def create_team_df(roster_dfs : dict[str, pd.DataFrame]):
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

    