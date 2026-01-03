"""
evaluation.py
Model evaluation, metrics, plotting, and cross-validation.

This module handles:
- Model performance metrics (MAE, RMSE, R²)
- Feature importance analysis
- Visualization/plotting
- Expanding window cross-validation for time series
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training import (
    load_data, merge_data, fill_season_to_date_with_baseline,
    prepare_features, train_model, TARGET
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info("MODEL EVALUATION")
    logger.info(f"MAE:  {metrics['mae']:.2f} fantasy points")
    logger.info(f"RMSE: {metrics['rmse']:.2f} fantasy points")
    logger.info(f"R²:   {metrics['r2']:.3f}")

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
    plot_residual_distribution(y_test, y_pred, output_dir / "residual_distribution.png")
    plot_error_by_fpts_range(y_test, y_pred, output_dir / "error_by_range.png")

    logger.info(f"All plots saved to {output_dir}/")


# =============================================================================
# CROSS-VALIDATION FOR TIME SERIES
# =============================================================================

def rolling_window_cv(
    game_logs_path: str | Path,
    players_path: str | Path,
    teams_path: str | Path,
    position_defense_path: Optional[str | Path] = None,
    start_season: int = 2020,
    end_season: int = 2024,
    window_size: int = 1,
    **xgb_params
) -> Dict:
    """
    Rolling window cross-validation for time series (RECOMMENDED for NBA/concept drift).

    Example with seasons 2022-2024, window_size=1:
        Fold 1: Train on [2022]      → Test on 2023
        Fold 2: Train on [2023]      → Test on 2024

    Example with window_size=2:
        Fold 1: Train on [2021, 2022] → Test on 2023
        Fold 2: Train on [2022, 2023] → Test on 2024

    **Better for NBA because:**
    - Play style evolves (3-point revolution, pace changes)
    - Rule changes affect scoring patterns
    - Recent data is more predictive than old data
    - Handles concept drift better than expanding window

    Args:
        game_logs_path: Path to game_logs_features.csv
        players_path: Path to processed_players.csv
        teams_path: Path to processed_teams.csv
        position_defense_path: Path to team_vs_position_defense.csv
        start_season: First season to use
        end_season: Last season (used only for testing in final fold)
        window_size: Number of seasons in training window (default=1)
        **xgb_params: Parameters to pass to XGBRegressor

    Returns:
        {
            'fold_metrics': [...],
            'avg_metrics': {'mae': ..., 'rmse': ..., 'r2': ...},
            'models': [...]
        }
    """
    logger.info("=" * 80)
    logger.info(f"ROLLING WINDOW CROSS-VALIDATION (window_size={window_size})")
    logger.info("=" * 80)

    # Load data once
    game_logs, players, teams, position_defense = load_data(
        game_logs_path, players_path, teams_path, position_defense_path
    )

    # Merge and prepare
    df = merge_data(game_logs, players, teams, position_defense)
    df = fill_season_to_date_with_baseline(df)

    # Prepare features
    X, y, feature_names = prepare_features(df)

    # Generate folds with rolling window
    test_seasons = list(range(start_season + window_size, end_season + 1))
    fold_metrics = []
    models = []

    for fold_idx, test_season in enumerate(test_seasons, start=1):
        # Rolling window: take last `window_size` seasons before test_season
        train_seasons = list(range(test_season - window_size, test_season))

        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold_idx}: Train on {train_seasons} → Test on {test_season}")
        logger.info(f"{'='*80}")

        # Split data
        train_mask = df['season'].isin(train_seasons)
        test_mask = df['season'] == test_season

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        logger.info(f"  Train: {len(X_train):,} samples")
        logger.info(f"  Test:  {len(X_test):,} samples")

        # Train model
        model = train_model(X_train, y_train)
        models.append(model)

        # Evaluate
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"  MAE:  {mae:.2f}")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²:   {r2:.3f}")

        fold_metrics.append({
            'fold': fold_idx,
            'train_seasons': train_seasons,
            'test_season': test_season,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test),
        })

    # Calculate average metrics
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])

    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Average MAE:  {avg_mae:.2f} ± {np.std([m['mae'] for m in fold_metrics]):.2f}")
    logger.info(f"Average RMSE: {avg_rmse:.2f} ± {np.std([m['rmse'] for m in fold_metrics]):.2f}")
    logger.info(f"Average R²:   {avg_r2:.3f} ± {np.std([m['r2'] for m in fold_metrics]):.3f}")
    logger.info(f"{'='*80}\n")

    return {
        'fold_metrics': fold_metrics,
        'avg_metrics': {
            'mae': avg_mae,
            'rmse': avg_rmse,
            'r2': avg_r2,
        },
        'models': models,
    }