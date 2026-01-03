"""
Merges processed data and trains XGBoost model for fantasy basketball prediction.

Data Flow:
    game_logs_features.csv (individual games + rolling stats)
        ├── LEFT JOIN processed_players.csv ON (player_normalized, season)
        └── LEFT JOIN processed_teams.csv ON (opponent, season)
            └── final_training_matrix → XGBoost
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Features
# Rolling features (from game_logs_features.csv)
ROLLING_FEATURES = [
    'pts_last_5', 'reb_last_5', 'ast_last_5',
    'stl_last_5', 'blk_last_5', 'tov_last_5',
    'min_last_5', 'fppg_last_3', 'fppg_last_5'# Removed 'fppg_last_5' and score went up a little
    'min_last_3', 'pts_last_3', 'reb_last_3',
    'ast_last_3', 'stl_last_3', 'blk_last_3', 'tov_last_3'
]

# Season-to-date features (from game_logs_features.csv - dynamic, no leakage)
SEASON_TO_DATE_FEATURES = [
    'pts_season_avg', 'reb_season_avg', 'ast_season_avg',
    'stl_season_avg', 'blk_season_avg', 'tov_season_avg',
    'fppg_season_avg', 'games_played_season',
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


# Data Loading and Merging
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

# Data Splitting
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
    **xgb_params
) -> XGBRegressor:
    """Train XGBoost regressor."""
    
    # Default parameters (reasonable for MVP)
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.001,
        'random_state': 42,
        'n_jobs': -1,
    }
    
    # Override with any provided params
    params = {**default_params, **xgb_params}
    
    logger.info(f"Training XGBoost with params: {params}")
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    logger.info("Training complete!")
    
    return model

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model performance."""
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
    }
    
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 50)
    logger.info(f"MAE:  {metrics['mae']:.2f} fantasy points")
    logger.info(f"RMSE: {metrics['rmse']:.2f} fantasy points")
    logger.info(f"R²:   {metrics['r2']:.3f}")
    logger.info("=" * 50)
    
    return metrics


def get_feature_importance(
    model: XGBRegressor,
    feature_names: List[str],
) -> pd.DataFrame:
    """Get feature importance ranking."""
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nFeature Importance:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance


def plot_feature_importance(
    importance: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    top_n: int = 15,
) -> None:
    """Plot horizontal bar chart of feature importance."""
    
    plt.figure(figsize=(10, 8))
    
    top_features = importance.head(top_n).sort_values('importance', ascending=True)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))
    
    plt.barh(top_features['feature'], top_features['importance'], color=colors)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_path}")
    
    plt.close()


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot actual vs predicted scatter plot with perfect prediction line."""
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, c='steelblue', label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics for annotation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.xlabel('Actual Fantasy Points', fontsize=12)
    plt.ylabel('Predicted Fantasy Points', fontsize=12)
    plt.title('Actual vs Predicted Fantasy Points', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    
    # Add metrics annotation
    textstr = f'MAE: {mae:.2f}\nR²: {r2:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.annotate(textstr, xy=(0.95, 0.05), xycoords='axes fraction', fontsize=11,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Actual vs predicted plot saved to {output_path}")
    
    plt.close()


def plot_residual_distribution(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot histogram of prediction residuals (errors)."""
    
    plt.figure(figsize=(10, 6))
    
    residuals = y_test - y_pred
    
    plt.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
    plt.axvline(x=residuals.mean(), color='orange', linestyle='-', lw=2, 
                label=f'Mean Error: {residuals.mean():.2f}')
    
    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend()
    
    # Add std annotation
    textstr = f'Std Dev: {residuals.std():.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.annotate(textstr, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=11,
                 verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Residual distribution plot saved to {output_path}")
    
    plt.close()


def plot_error_by_fpts_range(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot MAE across different fantasy point ranges (are we better at predicting stars vs bench?)."""
    
    plt.figure(figsize=(10, 6))
    
    # Create bins for fantasy point ranges
    bins = [0, 15, 25, 35, 45, 100]
    labels = ['0-15', '15-25', '25-35', '35-45', '45+']
    
    df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    df['range'] = pd.cut(df['actual'], bins=bins, labels=labels)
    
    # Calculate MAE for each range
    mae_by_range = df.groupby('range', observed=True).apply(
        lambda x: mean_absolute_error(x['actual'], x['predicted'])
    )
    counts = df.groupby('range', observed=True).size()
    
    x = range(len(labels))
    bars = plt.bar(x, mae_by_range.values, color='steelblue', edgecolor='white')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                 f'n={count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(x, labels)
    plt.xlabel('Actual Fantasy Points Range', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.title('Prediction Error by Fantasy Point Range', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error by range plot saved to {output_path}")
    
    plt.close()


def plot_all(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance: pd.DataFrame,
    output_dir: str | Path = "./plots",
) -> None:
    """Generate all visualization plots."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    logger.info("Generating plots...")
    
    plot_feature_importance(importance, output_dir / "feature_importance.png")
    plot_actual_vs_predicted(y_test, y_pred, output_dir / "actual_vs_predicted.png")
    plot_residual_distribution(y_test, y_pred, output_dir / "residual_distribution.png")
    plot_error_by_fpts_range(y_test, y_pred, output_dir / "error_by_range.png")
    
    logger.info(f"All plots saved to {output_dir}/")

def save_model(
    model: XGBRegressor,
    output_path: str | Path,
    feature_names: Optional[List[str]] = None,
) -> None:
    """Save model and feature names."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model using joblib (more reliable for sklearn-wrapped XGBoost)
    model_path = output_path.with_suffix('.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names for inference
    if feature_names:
        features_path = output_path.with_suffix('.features.txt')
        with open(features_path, 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"Feature names saved to {features_path}")


def load_model(model_path: str | Path) -> Tuple[XGBRegressor, List[str]]:
    """Load model and feature names."""
    
    model_path = Path(model_path)
    
    # Load model
    model = joblib.load(model_path.with_suffix('.joblib'))
    
    features_path = model_path.with_suffix('.features.txt')
    if features_path.exists():
        with open(features_path, 'r') as f:
            feature_names = f.read().strip().split('\n')
    else:
        feature_names = []
    
    return model, feature_names


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_pipeline(
    game_logs_path: str = "./data/processed/game_logs_features.csv",
    players_path: str = "./data/processed/processed_players.csv",
    teams_path: str = "./data/processed/processed_teams.csv",
    position_defense_path: str = "./data/processed/team_vs_position_defense.csv",
    model_output_path: str = "./models/fantasy_predictor",
    plots_output_dir: str = "./plots",
    test_season: int = 2024,
    generate_plots: bool = True,
) -> Tuple[XGBRegressor, dict, pd.DataFrame]:
    """
    Full training pipeline.
    
    Returns:
        model: Trained XGBoost model
        metrics: Evaluation metrics dict
        importance: Feature importance DataFrame
    """
    
    # 1. Load data
    game_logs, players, teams, position_defense = load_data(
        game_logs_path, players_path, teams_path, position_defense_path
    )
    
    # 2. Merge data
    df = merge_data(game_logs, players, teams, position_defense)

    # 2b. Fill season-to-date NaN with baseline (for early-season games)
    logger.info("Filling season-to-date NaN values with baseline stats...")
    df = fill_season_to_date_with_baseline(df)

    # 3. Prepare features
    X, y, feature_names = prepare_features(df)
    
    # 4. Train/test split
    X_train, X_test, y_train, y_test = split_by_season(
        df, X, y, test_season=test_season
    )
    
    # 5. Train model
    model = train_model(X_train, y_train)
    
    # 6. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # 7. Feature importance
    importance = get_feature_importance(model, feature_names)
    
    # 8. Generate plots
    if generate_plots:
        plot_all(model, X_test, y_test, importance, plots_output_dir)
    
    # 9. Save model
    save_model(model, model_output_path, feature_names)
    
    return model, metrics, importance


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fantasy basketball predictor")
    parser.add_argument("--data-dir", type=str, default="./data/processed",
                        help="Directory containing processed CSVs")
    parser.add_argument("--model-output", type=str, default="./models/fantasy_predictor",
                        help="Output path for trained model")
    parser.add_argument("--plots-dir", type=str, default="./plots",
                        help="Output directory for plots")
    parser.add_argument("--test-season", type=int, default=2024,
                        help="Season to use as test set")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    model, metrics, importance = train_pipeline(
        game_logs_path=data_dir / "game_logs_features.csv",
        players_path=data_dir / "processed_players.csv",
        teams_path=data_dir / "processed_teams.csv",
        position_defense_path=data_dir / "team_vs_position_defense.csv",
        model_output_path=args.model_output,
        plots_output_dir=args.plots_dir,
        test_season=args.test_season,
        generate_plots=not args.no_plots,
    )
    
    print("\n")
    print("Training Complete:")
    print("\n")
    print(f"\nModel saved to: {args.model_output}.joblib")
    if not args.no_plots:
        print(f"Plots saved to: {args.plots_dir}/")
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {metrics['mae']:.2f} fantasy points")
    print(f"  RMSE: {metrics['rmse']:.2f} fantasy points")
    print(f"  R²:   {metrics['r2']:.3f}")
    print(f"\nTop 5 Features:")
    print(importance.head(5).to_string(index=False))