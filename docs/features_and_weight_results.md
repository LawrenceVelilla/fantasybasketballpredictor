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
