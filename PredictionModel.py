import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def process_and_model_all_teams(csv_path):
    """
    Cleans data, splits into player volume groups, trains models per group,
    performs grid search for XGBoost, and prints evaluation metrics.
    """
    # Load and clean
    df = pd.read_csv(csv_path, header=0)
    df = df[df['Opponent'] != 'LAC']

    avg_fga = df.groupby('Player_ID')['FGA'].mean().reset_index()
    avg_fga.columns = ['Player_ID', 'avg_fga']
    df = df.merge(avg_fga, on='Player_ID', how='left')
    df['avg_fga_5g'] = df.groupby('Player_ID')['avg_fga_5g'].transform(lambda x: x.fillna(x.mean()))

    q1 = df['avg_fga_5g'].quantile(0)
    df['volume_group'] = pd.cut(df['avg_fga_5g'], bins=[-float('inf'), q1, float('inf')], labels=['low', 'high'])

    base_features = ['avg_pts_5g', 'avg_min_5g', 'avg_fga_5g', 'avg_fgm_5g', 'avg_fta_5g', 'avg_ftm_5g', 'std_fta_5g',
                     'pts_prev_game', 'pts_prev_game_per_min', 'pts_prev_game_per_min_squared', 'avg_fga_5g_squared',
                     'avg_fga_5g_avg_pts_5g', 'avg_fga_5g_avg_min_5g', 'ppg', 'ppg_squared', 'days_since_last_game',
                     'home_or_away', 'avg_efg_pct_2g', 'avg_ts_pct_2g', 'avg_usg_pct_2g', 'avg_off_rating_2g',
                     'avg_oreb_pct_3g', 'pie_prev_game', 'OPP_BOX_OUTS', 'OPP_DREB', 'OPP_BLK', 'OPP_DEFLECTIONS',
                     'OPP_FG_PCT_RANK', 'Position_Guard-Forward', 'Position_Center', 'Position_Guard',
                     'Position_Forward', 'START_POSITION', 'weighted_avg_pts_3g', 'weighted_avg_min_3g', 'won_prev_game', 
                     'usg_min_product', 'pie_fga_product', 'OPP_W_PCT_RANK', 
                     'off_def_rating_diff', 'momentum_positive', 'pts_momentum_2g', 'law_of_averages_ratio',
                      'fga_vs_opp_def']

    for group in ['high', 'low']:
        print(f"\n--- {group.upper()} VOLUME PLAYERS ---")

        group_df = df[df['volume_group'] == group]
        print("Row count before dropping nulls", group_df.shape[0])
        group_df = group_df.dropna(subset=base_features + ['PTS'])
        print("Row count after dropping nulls", group_df.shape[0])

        if group_df.shape[0] < 30:
            print("Not enough data to train.")
            continue

        X = group_df[base_features]
        y = group_df['PTS']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # --- Random Forest ---
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        print("Random Forest → MAE: {:.2f}, R²: {:.2f}".format(
            mean_absolute_error(y_test, rf_pred),
            r2_score(y_test, rf_pred)
        ))

        # --- XGBoost with Grid Search ---
        xgb = XGBRegressor(objective='reg:squarederror', random_state=42)


        param_grid = {
            'n_estimators': [100, 300,600],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4],
            'subsample': [0.7],
            'colsample_bytree': [1.0],
            'gamma': [0, 5, 10],
            'min_child_weight': [3, 6]
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(estimator=xgb,
                                   param_grid=param_grid,
                                   scoring='r2',
                                   cv=cv,
                                   verbose=1,
                                   n_jobs=-1)

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        xgb_pred = best_model.predict(X_test)

        print("XGBoost (Tuned) → MAE: {:.2f}, R²: {:.2f}".format(
            mean_absolute_error(y_test, xgb_pred),
            r2_score(y_test, xgb_pred)
        ))

        print("Best Params:", grid_search.best_params_)
