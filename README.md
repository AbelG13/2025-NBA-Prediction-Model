# 🏀 NBA Player Points Prediction Model (2025 Season)

This project builds a machine learning model using data from the 2025 NBA regular season to **predict how many points a player will score in an upcoming game**, aiming to have a reliable system ready for the **2025 NBA Playoffs**.

## 📌 Project Goal

Create a robust, near-real-time prediction model that can:
- Analyze player and team trends
- Engineer predictive features from game logs and opponent tendencies
- Forecast player point totals using only historical and contextual data

---

## 🧠 Project Overview

This project uses the [`nba_api`](https://github.com/swar/nba_api) to collect real NBA game log and advanced box score data, which is processed into model-ready features for training.

Key aspects:
- Player performance across recent games
- Opponent defensive metrics
- Game-specific context (e.g., home/away splits)

The model has transitioned to **XGBoost** for its superior handling of non-linear feature interactions, regularization capabilities, and robustness against overfitting, outperforming previous Random Forest models.

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `AddPositions.py` | Adds player position information to the dataset |
| `AdvancedCSV.py` | Gathers and cleans advanced box score statistics |
| `CreateFeatures.py` | Builds rolling averages, interaction terms, and engineered features for each player |
| `FeatureStrengthTest.py` | Tests the correlation and importance of engineered features |
| `PredictionModel.py` | Trains the XGBoost regression model and tunes it using cross-validation and grid search |
| `main.py` | Main script to prepare data, train the model, and evaluate results |

---

## 🧪 Current Results

- ✅ Data leakage eliminated by using only historical information when predicting
- ✅ Feature engineering expanded with weighted rolling averages, opponent-adjusted metrics, and interaction terms
- ✅ Switched to XGBoost, tuned hyperparameters, and achieved a **current R² of 0.74**
- ⚠️ Major Challenge: The `nba_api` data appears to update only **monthly**, not daily, making it difficult to maintain live predictions without manual intervention

---

## 🔮 Next Steps

- Develop strategies to supplement or verify live data sources beyond `nba_api`
- Further refine feature creation, particularly opponent adjustment features
- Expand model evaluation on playoff game logs once data becomes available

---

## 💡 Why This Matters

Accurate player scoring predictions power:
- DFS (daily fantasy sports) optimization
- NBA prop betting models
- Realistic game simulations
- Sports journalism, fan engagement, and analytics products

---
