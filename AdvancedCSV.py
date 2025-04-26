import pandas as pd
import time
from nba_api.stats.endpoints import BoxScoreAdvancedV2, commonplayerinfo
import os

def advanced_stats():
    """
    Create CSV with advanced stats and player position.
    """
    df = pd.read_csv('all_teams.csv')
    game_ids = df['Game_ID'].drop_duplicates().tolist()
    
    # prepend 00 onto every ID in game_ids
    game_ids = [f'00{game_id}' for game_id in game_ids]

    curr_df = pd.read_csv('advanced_stats.csv', header=0)

    # Remove all game_ids already in the curr_df from the game_ids list
    game_ids = [game_id for game_id in game_ids if game_id not in curr_df['Game_ID'].tolist()]
    print(len(game_ids))

    df_advanced = pd.DataFrame()

    sleep_count=0
    e_count = 0
    if len(game_ids) > 0:
        for game_id in game_ids:
            try:
                response = BoxScoreAdvancedV2(game_id=game_id)
                data = response.get_data_frames()
                if len(data) == 0:
                    print(f"No data for game ID {game_id}")
                    continue
                box_score = data[0]
                
                box_score = box_score.rename(columns={'PLAYER_ID': 'Player_ID', 'GAME_ID': 'Game_ID'})
                # List of all player IDS in all_team.csv
                df_qual = pd.read_csv('all_teams.csv')
                player_ids = df_qual['Player_ID'].drop_duplicates().tolist()

                # Filter box_score to only include player_ids 
                box_score = box_score[box_score['Player_ID'].isin(player_ids)] 

                # prepend 00 onto every ID in box_score
                box_score['Game_ID'] = '00' + box_score['Game_ID'].astype(str)

                df_advanced = pd.concat([df_advanced, box_score], ignore_index=True)
                print(f"Success {game_id}")
            except Exception as e:
                e_count += 1
                print(f"Error fetching data for Game ID {game_id}: {e}")
                if e_count>= 2:
                    break
            sleep_count += 1
            if sleep_count%5==0:
                time.sleep(2)  # to avoid rate limit
        df_advanced.to_csv('advanced_stats.csv', mode='a', index=False, header=not os.path.exists('advanced_stats.csv'))

    else:
        return
