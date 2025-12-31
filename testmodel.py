import pandas as pd



# Make this editable later
FANTASY_SETTINGS = {
    'pts': 1,
    'oreb': 1.5,
    'dreb': 1,
    'ast': 1.5,
    'stl': 2.5,
    'blk': 2.5,
    'to': -1.5,
    '3pm': 1,
    'ftm': 0.5,
    'ftmi': -0.5,
    'fgm': 1,
    'fgmi': -1,
    'tw': 1
}

# Design df.
# player_name | team | pos | age | g | gs | mp | fg | fga | fg% | 3p | 3pa | 3p% | ft | fta | ft% | orb | drb | trb | ast | stl | blk | tov | pf | pts | fantasy_pts
# team_name | pace | efg% | ts% | usg% | bbr_pos | season_year | ppg | rpg | apg | spg | bpg | topg 

# The features the model learns from
FEATURES = [
    # 1. Base Stats (Current)
    'pts_avg', 'ast_avg', 'reb_avg', 'stl_avg', 'blk_avg', 
    'tov_avg', 'min_avg', 'usg_pct',
    # 2. Historical (Lag)
    'pts_prev', 'min_prev', 'fppg_prev',
    # 3. Recent Form
    'pts_last_5', 'min_last_5', 'fppg_last_5', 
    # 4. Context
    'opp_def_rating', 'opp_pace', 'is_home', 'rest_days'
]

# The Target (What we are trying to guess)
TARGET = 'actual_fantasy_points_scored'