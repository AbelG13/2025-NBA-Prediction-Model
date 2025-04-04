from FeatureStrengthTest import analyze_feature_strength
from CreateTeamDF import create_team_df
from QualifyingPlayers import qualifying_players
from PredictionModel import train_and_test_model


team = 'Houston Rockets'

# Test model
x = train_and_test_model(team)
print(x)

# Test feature strength
df = create_team_df(qualifying_players(team))
analyze_feature_strength(df)