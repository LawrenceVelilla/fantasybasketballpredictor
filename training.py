"""
training.py
Data preparation and model training for fantasy basketball predictions.

This module handles:
- Loading and merging data sources
- Feature preparation
- Model training and saving
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import joblib
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rolling features (from game_logs_features.csv)
ROLLING_FEATURES = [
    'pts_last_5', 'reb_last_5', 'ast_last_5',
    'stl_last_5', 'blk_last_5', 'tov_last_5',
    'min_last_5', 'fppg_last_3', 'fppg_last_5',
    'min_last_3', 'pts_last_3', 'reb_last_3',
    'ast_last_3', 'stl_last_3', 'blk_last_3', 'tov_last_3',
    'usage_rate_last_3', 'usage_rate_last_5',
]

# Season-to-date features (from game_logs_features.csv - dynamic, no leakage)
SEASON_TO_DATE_FEATURES = [
    'pts_season_avg', 'reb_season_avg', 'ast_season_avg',
    'stl_season_avg', 'blk_season_avg', 'tov_season_avg', 'min_season_avg',
    'fppg_season_avg', 'games_played_season', 'usage_rate'
]

# Baseline features (from processed_players.csv - last season's stats)
# Used as fallback for early-season games when season-to-date is unavailable
BASELINE_FEATURES = [
    'pts_per_game', 'ast_per_game', 'trb_per_game',
    'stl_per_game', 'blk_per_game', 'tov_per_game',
    'fg_percent', 'fppg',
]

# Matchup features (from processed_teams.csv + position_defense.csv)
MATCHUP_FEATURES = [
    'team_pace',   # player's team pace (more possessions = more opportunities)
    'opp_drtg',    # opponent defensive rating (easier defense = more points)
    'opp_pace',    # opponent pace (affects game tempo)
    'opp_pos_fg_pct',   # opponent FG% allowed to player's position
    'opp_pos_fg_diff',  # opponent FG% diff vs league avg for position
]

# Situational features (from game_logs_features.csv)
SITUATION_FEATURES = [
    'is_back_to_back',
]

# Position features (one-hot encoded from processed_players.csv)
POSITION_FEATURES = [
    'is_pg', 'is_sg', 'is_sf', 'is_pf', 'is_c',
]

# All features combined (season-to-date replaces baseline as primary, baseline used for fallback)
ALL_FEATURES = ROLLING_FEATURES + SEASON_TO_DATE_FEATURES + MATCHUP_FEATURES + SITUATION_FEATURES + POSITION_FEATURES

# Target variable
TARGET = 'actual_fantasy_pts'


def load_data(
    game_logs_path: str | Path,
    players_path: str | Path,
    teams_path: str | Path,
    position_defense_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load all processed CSVs."""

    logger.info(f"Loading game logs from {game_logs_path}")
    game_logs = pd.read_csv(game_logs_path)
    logger.info(f"- {len(game_logs)} game logs")

    logger.info(f"Loading player baselines from {players_path}")
    players = pd.read_csv(players_path)
    logger.info(f"- {len(players)} player-seasons")

    logger.info(f"Loading team stats from {teams_path}")
    teams = pd.read_csv(teams_path)
    logger.info(f"- {len(teams)} team-seasons")

    position_defense = None
    if position_defense_path and Path(position_defense_path).exists():
        logger.info(f"Loading position defense from {position_defense_path}")
        position_defense = pd.read_csv(position_defense_path)
        logger.info(f"- {len(position_defense)} position-team-seasons")
    else:
        logger.info("Position defense data not found, skipping")

    return game_logs, players, teams, position_defense


def merge_data(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    position_defense: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge all data sources into final training matrix.

    Join 1: game_logs + players ON (player_normalized, season)
    Join 2: result + teams ON (team, season) for player's team pace
    Join 3: result + teams ON (opponent, season) for opponent stats
    Join 4: result + position_defense ON (opponent, season, position) for position matchup
    """

    # Filter game_logs to only seasons where we have player/team data
    min_player_season = players['season'].min()
    min_team_season = teams['season'].min()
    min_available_season = max(min_player_season, min_team_season)

    original_count = len(game_logs)
    game_logs = game_logs[game_logs['season'] >= min_available_season].copy()
    filtered_count = len(game_logs)

    logger.info(f"Filtered game logs to seasons >= {min_available_season}")
    logger.info(f"  → Kept {filtered_count:,} of {original_count:,} game logs ({100*filtered_count/original_count:.1f}%)")

    logger.info("Merging game logs with player baselines...")

    # Select only needed columns from players to avoid duplicates
    player_cols = ['player_normalized', 'season', 'pos'] + [
        c for c in BASELINE_FEATURES if c in players.columns
    ]
    player_cols = [c for c in player_cols if c in players.columns]
    players_subset = players[player_cols].copy()

    # Remove duplicates in players (keep first)
    players_subset = players_subset.drop_duplicates(
        subset=['player_normalized', 'season'],
        keep='first'
    )

    # JOIN 1: Add player baselines (including position)
    df = game_logs.merge(
        players_subset,
        on=['player_normalized', 'season'],
        how='left',
        suffixes=('', '_baseline')
    )

    logger.info(f"  → After player join: {len(df)} rows")

    # Normalize position to primary position (first position if multi-position)
    if 'pos' in df.columns:
        df['primary_pos'] = df['pos'].astype(str).str.split('-').str[0].str.strip()
        # Map common variations
        pos_map = {
            'PG': 'PG', 'SG': 'SG', 'SF': 'SF', 'PF': 'PF', 'C': 'C',
            'G': 'PG', 'F': 'SF', 'G-F': 'SG', 'F-G': 'SF', 'F-C': 'PF', 'C-F': 'C'
        }
        df['primary_pos'] = df['primary_pos'].map(pos_map)  # Unknown positions become NaN

        # One-hot encode position (NaN/unknown = all zeros)
        df['is_pg'] = (df['primary_pos'] == 'PG').astype(int)
        df['is_sg'] = (df['primary_pos'] == 'SG').astype(int)
        df['is_sf'] = (df['primary_pos'] == 'SF').astype(int)
        df['is_pf'] = (df['primary_pos'] == 'PF').astype(int)
        df['is_c'] = (df['primary_pos'] == 'C').astype(int)

        # Fill NaN for position defense join (use SF as fallback for join only)
        df['primary_pos'] = df['primary_pos'].fillna('SF')
    else:
        # No position data available - create columns with zeros
        df['primary_pos'] = 'SF'
        df['is_pg'] = 0
        df['is_sg'] = 0
        df['is_sf'] = 0
        df['is_pf'] = 0
        df['is_c'] = 0

    # JOIN 2: Add player's TEAM stats (pace affects their opportunities)
    logger.info("Merging with player's team stats...")

    teams_player = teams.rename(columns={
        'team_drtg': 'team_drtg',
        'team_pace': 'team_pace',
        'team_ortg': 'team_ortg',
    })

    team_cols = ['team', 'season', 'team_pace', 'team_ortg']
    team_cols = [c for c in team_cols if c in teams_player.columns]
    teams_player_subset = teams_player[team_cols].copy()

    df = df.merge(
        teams_player_subset,
        on=['team', 'season'],
        how='left'
    )

    logger.info(f"- After team join: {len(df)} rows")

    # JOIN 3: Add OPPONENT stats
    logger.info("Merging with opponent team stats...")

    teams_opp = teams.rename(columns={
        'team': 'opponent',
        'team_drtg': 'opp_drtg',
        'team_pace': 'opp_pace',
        'team_ortg': 'opp_ortg',
        'team_pts_per_game': 'opp_pts_per_game',
    })

    opp_cols = ['opponent', 'season', 'opp_drtg', 'opp_pace']
    opp_cols = [c for c in opp_cols if c in teams_opp.columns]
    teams_opp_subset = teams_opp[opp_cols].copy()

    df = df.merge(
        teams_opp_subset,
        on=['opponent', 'season'],
        how='left'
    )

    logger.info(f"- After opponent join: {len(df)} rows")

    # JOIN 4: Add POSITION DEFENSE stats
    if position_defense is not None and 'primary_pos' in df.columns:
        logger.info("Merging with position defense stats...")

        # Prepare position defense for join
        pos_def = position_defense.copy()
        pos_def = pos_def.rename(columns={'team': 'opponent', 'position': 'primary_pos'})

        pos_def_cols = ['opponent', 'season', 'primary_pos', 'opp_pos_fg_pct', 'opp_pos_fg_diff']
        pos_def_cols = [c for c in pos_def_cols if c in pos_def.columns]

        if len(pos_def_cols) >= 3:  # Need at least join keys + 1 feature
            pos_def_subset = pos_def[pos_def_cols].copy()

            df = df.merge(
                pos_def_subset,
                on=['opponent', 'season', 'primary_pos'],
                how='left'
            )

            logger.info(f"  → After position defense join: {len(df)} rows")

    return df


def fill_season_to_date_with_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill season-to-date NaN values with baseline (last season) stats.

    For early-season games where season-to-date stats are NaN (no prior games),
    we fall back to the player's stats from the previous season.

    Mapping:
        pts_season_avg  <- pts_per_game (baseline)
        reb_season_avg  <- trb_per_game (baseline)
        ast_season_avg  <- ast_per_game (baseline)
        stl_season_avg  <- stl_per_game (baseline)
        blk_season_avg  <- blk_per_game (baseline)
        tov_season_avg  <- tov_per_game (baseline)
        fppg_season_avg <- fppg (baseline)
    """
    df = df.copy()

    # Mapping from season-to-date column to baseline fallback column
    fallback_map = {
        'pts_season_avg': 'pts_per_game',
        'reb_season_avg': 'trb_per_game',
        'ast_season_avg': 'ast_per_game',
        'stl_season_avg': 'stl_per_game',
        'blk_season_avg': 'blk_per_game',
        'tov_season_avg': 'tov_per_game',
        'fppg_season_avg': 'fppg',
    }

    filled_count = 0
    for season_col, baseline_col in fallback_map.items():
        if season_col in df.columns and baseline_col in df.columns:
            # Count NaNs before fill
            nan_before = df[season_col].isna().sum()

            # Fill NaN in season-to-date with baseline value
            df[season_col] = df[season_col].fillna(df[baseline_col])

            nan_after = df[season_col].isna().sum()
            filled = nan_before - nan_after

            if filled > 0:
                filled_count += filled
                logger.info(f"  Filled {filled:,} NaN in {season_col} with {baseline_col}")

    if filled_count > 0:
        logger.info(f"Total season-to-date values filled from baseline: {filled_count:,}")

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str] = ALL_FEATURES,
    target_col: str = TARGET,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix X and target vector y.

    Returns:
        X: Feature DataFrame
        y: Target Series
        available_features: List of features actually present
    """

    # Check which features are available
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]

    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    logger.info(f"Using {len(available_features)} features: {available_features}")

    # Extract X and y
    X = df[available_features].copy()
    y = df[target_col].copy()

    # Log missing value stats
    missing_pct = (X.isnull().sum() / len(X) * 100).round(2)
    if missing_pct.any():
        logger.info("Missing value percentages:")
        for col, pct in missing_pct[missing_pct > 0].items():
            logger.info(f"  {col}: {pct}%")

    return X, y, available_features

def split_by_season(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_season: int = 2024,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data by season (temporal split to avoid leakage).

    Train: all seasons before test_season
    Test: test_season only
    """

    train_set = df['season'] < test_season
    test_set = df['season'] == test_season

    X_train, X_test = X[train_set], X[test_set]
    y_train, y_test = y[train_set], y[test_set]

    logger.info(f"Train set: {len(X_train)} samples (seasons < {test_season})")
    logger.info(f"Test set: {len(X_test)} samples (season = {test_season})")

    return X_train, X_test, y_train, y_test

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: Optional[np.ndarray] = None,
) -> XGBRegressor:
    """Train XGBoost regressor with regularization."""

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,      # 70% of features per tree (was 0.001 which was horrendous) --> Controls how many features are seen by each tree
        min_child_weight=15,         # Minimum sum of instance weight in child
        gamma=0.1,                  # Minimum loss reduction for split
        reg_alpha=0.1,              # L1 regularization on weights
        reg_lambda=3,             # L2 regularization on weights 
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    logger.info("Training complete!")

    return model

def save_model(
    model: XGBRegressor,
    output_path: str | Path,
    feature_names: Optional[List[str]] = None,
) -> None:
    """Save model in multiple formats and feature names."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model using joblib (more reliable for sklearn-wrapped XGBoost)
    model_path = output_path.with_suffix('.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save model as JSON (for cross-platform compatibility using the underlying booster)
    json_path = output_path.with_suffix('.json')
    model.get_booster().save_model(json_path)
    logger.info(f"Model saved to {json_path}")

    # Save feature names for inference
    if feature_names:
        features_path = output_path.with_suffix('.features.txt')
        with open(features_path, 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"Feature names saved to {features_path}")


if __name__ == "__main__":
    """Train model on all available seasons for production use."""

    logger.info("=" * 80)
    logger.info("TRAINING MODEL ON ALL SEASONS (PRODUCTION)")
    logger.info("=" * 80)

    # Define paths
    DATA_DIR = Path("data/processed")
    GAME_LOGS_PATH = DATA_DIR / "game_logs_features.csv"
    PLAYERS_PATH = DATA_DIR / "processed_players.csv"
    TEAMS_PATH = DATA_DIR / "processed_teams.csv"
    POSITION_DEFENSE_PATH = DATA_DIR / "team_vs_position_defense.csv"
    MODEL_OUTPUT_PATH = Path("models/fantasy_predictor.joblib")

    # Load data
    logger.info("\n1. Loading data...")
    game_logs, players, teams, position_defense = load_data(
        game_logs_path=GAME_LOGS_PATH,
        players_path=PLAYERS_PATH,
        teams_path=TEAMS_PATH,
        position_defense_path=POSITION_DEFENSE_PATH,
    )

    # Merge data
    logger.info("\n2. Merging data sources...")
    df = merge_data(game_logs, players, teams, position_defense)

    # Fill season-to-date NaNs with baseline
    logger.info("\n3. Filling missing season-to-date values with baseline stats...")
    df = fill_season_to_date_with_baseline(df)

    # Prepare features
    logger.info("\n4. Preparing features...")
    X, y, available_features = prepare_features(df)

    # Drop rows with NaN in features or target
    # valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    # X_clean = X[valid_mask]
    # y_clean = y[valid_mask]

 
    # Train model on ALL data
    logger.info("\n5. Training model...")
    model = train_model(X, y)

    # Save model
    save_model(model, MODEL_OUTPUT_PATH, available_features)
