import os
import time
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

def add_position():
    df_advanced = pd.read_csv('advanced_stats.csv')

    player_ids = df_advanced['Player_ID'].drop_duplicates().tolist()

    positions = {}
    count = 0
    for id in player_ids:
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=id)
            position = info.get_data_frames()[0].loc[0, 'POSITION']
            positions[id] = position
            print(f"Success {id}")
        except Exception as e:
            positions[id] = "Unknown"
            print(f"Error fetching position for Player ID {id}: {e}")
        count += 1
        if count % 5 == 0:
            time.sleep(2)

    # Map position to each row using Player_ID
    df_advanced['Position'] = df_advanced['Player_ID'].apply(lambda x: positions.get(x, "Unknown"))

    # Overwrite CSV with new column
    df_advanced.to_csv('advanced_stats.csv', index=False)
