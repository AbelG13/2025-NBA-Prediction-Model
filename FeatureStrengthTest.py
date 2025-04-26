import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- CONFIGURABLE PARAMS ---
TARGET = 'PTS'
FEATURE_COLUMNS = ['avg_pts_5g', 'avg_min_5g', 'avg_fga_5g', 'avg_fgm_5g', 'avg_fta_5g', 'avg_ftm_5g', 'std_fta_5g','pts_prev_game',
    'pts_prev_game_per_min', 'pts_prev_game_per_min_squared', 'avg_fga_5g_squared', 'avg_fga_5g_avg_pts_5g',
    'avg_fga_5g_avg_min_5g', 'ppg', 'ppg_squared', 'days_since_last_game', 'home_or_away', 
    'avg_efg_pct_2g','avg_ts_pct_2g', 'avg_usg_pct_2g',  'avg_off_rating_2g','avg_oreb_pct_3g','pie_prev_game','OPP_BOX_OUTS', 
    'OPP_DREB', 'OPP_BLK', 'OPP_DEFLECTIONS','OPP_FG_PCT_RANK', 'Position_Guard-Forward', 
    'Position_Center', 'Position_Guard','Position_Forward', 'START_POSITION' ]

# --- MAIN FUNCTION ---
def analyze_feature_strength(df):
    """
    Analyze the strength of features in a dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze.
    """

    # Drop rows with missing values in feature columns or target
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET])
    
    # Correlation Matrix
    print("\n--- Feature Correlation with Target ---")
    corr = df[FEATURE_COLUMNS + [TARGET]].corr()[TARGET].sort_values(ascending=False)
    print(corr)

    # Plot Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[FEATURE_COLUMNS + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Train Random Forest
    print("\n--- Random Forest Feature Importances ---")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    print(importances.sort_values(ascending=False))

    # Plot Importances
    importances.plot(kind='barh', figsize=(10, 6), title='Random Forest Feature Importances')
    plt.xlabel("Importance")
    plt.show()

    # Train Linear Model for baseline R^2
    print("\n--- Linear Regression Model Summary ---")
    lm = LinearRegression()
    lm.fit(X, y)
    y_pred = lm.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Linear Regression R^2: {r2:.4f}")


