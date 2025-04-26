# 🏀 NBA Player Points Prediction Model (2025 Season)

This project is focused on building a machine learning model using data from the 2025 NBA regular season to **predict how many points a player will score in an upcoming game** with the goal of having a reliable system ready to deploy for the **2025 NBA Playoffs**.

## 📌 Project Goal

To create a robust, real-time model that can:
- Analyze player and team trends
- Engineer predictive features from game logs and opponent tendencies
- Forecast player point totals using historical and contextual data

---

## 🧠 Project Overview

This project uses the [`nba_api`](https://github.com/swar/nba_api) to gather real NBA game log data and engineer features for training a machine learning model.

The model is trained to predict `PTS` (points scored in a game) for each player on a given team using:
- Their **last 5 games' performance**
- Opponent **defensive metrics**
- Game context (e.g., home/away, rest days)

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `QualifyingPlayers.py` | Identifies players who qualify based on minutes played in recent games |
| `CreateFeatures.py` | Builds rolling stat features (e.g., avg points over last 5 games) for a given player |
| `CreateTeamDF.py` | Assembles a full training DataFrame for all qualifying players on a team |
| `FeatureStrengthTest.py` | Tests correlation and importance of features against the target variable (`PTS`) |
| `PredictionModel.py` | Trains a `RandomForestRegressor` and evaluates performance (MAE, R²) |
| `main.py` | Main entry point for running model training and feature tests |

---

## 🧪 Current Results

- ✅ Data leakage removed by only using past games to predict future ones
- ✅ Interaction terms (e.g., `avg_fga_5g * avg_pts_5g`) and non-linear transforms (e.g., `avg_fga_5g²`) added
- 📉 Model currently far below desired R² value, indicating room for feature improvement

---

## 🔮 Next Steps

- Improve feature engineering with deeper contextual stats
- Test additional models: LightGBM, XGBoost, Ridge regression
- Incorporate betting odds and Vegas totals
- Optimize for live game-day prediction use

---

## 💡 Why This Matters

Accurately forecasting player scoring has applications in:
- DFS (daily fantasy sports)
- NBA prop betting
- Game simulations & commentary
- Data journalism and fan engagement

---
