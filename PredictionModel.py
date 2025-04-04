from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from CreateTeamDF import create_team_df
from QualifyingPlayers import qualifying_players

# --- CONFIG ---
TARGET_STAT = 'PTS' 

FEATURE_COLUMNS = ['avg_fgm_5g', 'avg_ftm_5g', 'avg_fta_5g', 'avg_pts_5g', 'avg_min_5g', 
'avg_fga_5g', 'std_fta_5g', 'OPP_DREB', 'OPP_BLK','home_or_away',
'OPP_DEFLECTIONS','OPP_FG_PCT_RANK', 'avg_fga_5g_squared','avg_fga_5g_avg_pts_5g',
'avg_fga_5g_avg_min_5g', 'std_fta_5g', 'OPP_DREB', 'OPP_BLK', 'OPP_DEFLECTIONS','OPP_FG_PCT_RANK']


# --- TRAIN / TEST MODEL ---
def train_and_test_model(team):
    """ 
    Train and test a model for a specific team. Print MAE and R^2.
    
    Args:
        team (str): The team name.
    
    Returns:
        model: The trained model.
    """

    df = create_team_df(qualifying_players(team))  

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_STAT]

    # Time-aware split: Use latest games as test set
    df_sorted = df.sort_values(by='GAME_DATE')
    split_idx = int(0.8 * len(df_sorted))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}, R^2: {r2:.2f}")

    return model
