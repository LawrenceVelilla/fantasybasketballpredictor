(ROLLING_FEATURES = [
    'pts_last_5', 'reb_last_5', 'ast_last_5',
    'stl_last_5', 'blk_last_5', 'tov_last_5',
    'min_last_5', 'fppg_last_3', 'fppg_last_5',
    'min_last_3', 'pts_last_3', 'reb_last_3',
    'ast_last_3', 'stl_last_3', 'blk_last_3', 'tov_last_3',
    'usage_rate_last_3', 'usage_rate_last_5',
]

SEASON_TO_DATE_FEATURES = [
    'pts_season_avg', 'reb_season_avg', 'ast_season_avg',
    'stl_season_avg', 'blk_season_avg', 'tov_season_avg', 'min_season_avg',
    'fppg_season_avg', 'games_played_season', 'usage_rate'
]

BASELINE_FEATURES = [
    'pts_per_game', 'ast_per_game', 'trb_per_game',
    'stl_per_game', 'blk_per_game', 'tov_per_game',
    'fg_percent', 'fppg',
]

MATCHUP_FEATURES = [
    'team_pace',   # player's team pace (more possessions = more opportunities)
    'opp_drtg',    # opponent defensive rating (easier defense = more points)
    'opp_pace',    # opponent pace (affects game tempo)
    'opp_pos_fg_pct',   # opponent FG% allowed to player's position
    'opp_pos_fg_diff',  # opponent FG% diff vs league avg for position
]

SITUATION_FEATURES = [
    'is_back_to_back',
]

POSITION_FEATURES = [
    'is_pg', 'is_sg', 'is_sf', 'is_pf', 'is_c',
]

default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.001,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
    }
) --> (MAE:  7.09 fantasy points - RMSE: 9.10 fantasy points - R²: 0.581)


model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,      # 70% of features per tree (was 0.001 which was horrendous) --> Controls how many features are seen by each tree
        min_child_weight=12,         # Minimum sum of instance weight in child
        gamma=0.1,                  # Minimum loss reduction for split
        reg_alpha=0.1,              # L1 regularization on weights
        reg_lambda=1.0,             # L2 regularization on weights (default but explicit)
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    ) -->   Average Performance Across Folds:
  MAE:  6.58 fantasy points
  RMSE: 8.55 fantasy points
  R²:   0.629

Fold-by-Fold Results:
  Fold 1: Train [2020] → Test 2021
    MAE: 6.56, RMSE: 8.51, R²: 0.612
  Fold 2: Train [2021] → Test 2022
    MAE: 6.60, RMSE: 8.55, R²: 0.630
  Fold 3: Train [2022] → Test 2023
    MAE: 6.56, RMSE: 8.53, R²: 0.651
  Fold 4: Train [2023] → Test 2024
    MAE: 6.62, RMSE: 8.63, R²: 0.624



model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,      # 70% of features per tree (was 0.001 which was horrendous) --> Controls how many features are seen by each tree
        min_child_weight=15,         # Minimum sum of instance weight in child
        gamma=0.1,                  # Minimum loss reduction for split
        reg_alpha=0,              # L1 regularization on weights
        reg_lambda=3.0,             # L2 regularization on weights (default but explicit)
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    ) --> 



Test Set Performance:
  MAE:  6.57 fantasy points
  RMSE: 8.55 fantasy points
  R²:   0.630

Top 5 Features:
        feature  importance
fppg_season_avg    0.462321
    fppg_last_5    0.302383
    fppg_last_3    0.044709
     usage_rate    0.042493
     min_last_3    0.032244