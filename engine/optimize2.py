"""
optimize.py
Hyperparameter optimization for fantasy basketball predictor.

Finds the best XGBoost parameters using season-based rolling window CV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import logging
from itertools import product

from training import (
    load_data, merge_data, fill_season_to_date_with_baseline,
    prepare_features, train_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    start_season: int = 2022,
    end_season: int = 2024,
    window_size: int = 1,
) -> dict:
    """
    Run grid search using rolling window CV by season (like your rolling_window_cv).

    Example with window_size=1:
        Fold 1: Train on [2021] → Test on 2022
        Fold 2: Train on [2022] → Test on 2023

    This respects season boundaries and handles NBA concept drift.

    Returns:
        {
            'best_params': {...},
            'best_score': float,
            'all_results': [...]
        }
    """

    logger.info("Starting hyperparameter optimization with rolling window CV...")
    logger.info(f"Window size: {window_size} season(s)")
    logger.info(f"Testing {np.prod([len(v) for v in param_grid.values()])} parameter combinations")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    logger.info(f"Total combinations: {len(param_combinations)}")

    # Test seasons for rolling window
    test_seasons = list(range(start_season + window_size, end_season + 1))
    logger.info(f"Test seasons: {test_seasons}")

    # Store results for all parameter combinations
    all_results = []

    for param_combo in param_combinations:
        params = dict(zip(param_names, param_combo))
        logger.info(f"\nTesting params: {params}")

        fold_scores = []

        # Rolling window CV
        for test_season in test_seasons:
            train_seasons = list(range(test_season - window_size, test_season))

            # Split data by season
            train_mask = df['season'].isin(train_seasons)
            test_mask = df['season'] == test_season

            X_train_fold = X[train_mask]
            y_train_fold = y[train_mask]
            X_test_fold = X[test_mask]
            y_test_fold = y[test_mask]

            # Train model with current params
            model = XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
                **params  # Override with params being tested
            )

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            r2 = r2_score(y_test_fold, y_pred)

            fold_scores.append(r2)

        # Average R² across folds
        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)

        logger.info(f"  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")

        all_results.append({
            'params': params,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'fold_scores': fold_scores
        })

    # Find best parameters
    best_result = max(all_results, key=lambda x: x['mean_r2'])

    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Best parameters: {best_result['params']}")
    logger.info(f"Best CV R² score: {best_result['mean_r2']:.4f} ± {best_result['std_r2']:.4f}")
    logger.info(f"{'='*80}")

    return {
        'best_params': best_result['params'],
        'best_score': best_result['mean_r2'],
        'all_results': all_results
    }


def plot_param_performance(results: dict, param_name: str, output_path: str = None):
    """Plot parameter value vs performance."""

    all_results = results['all_results']

    # Extract results for this parameter
    data = []
    for result in all_results:
        if param_name in result['params']:
            data.append({
                'param': result['params'][param_name],
                'score': result['mean_r2'],
                'std': result['std_r2']
            })

    if not data:
        logger.warning(f"Parameter {param_name} not found in results")
        return

    df = pd.DataFrame(data)

    # Group by parameter value (average across other params)
    grouped = df.groupby('param').agg({
        'score': 'mean',
        'std': 'mean'
    }).reset_index().sort_values('param')

    plt.figure(figsize=(10, 6))

    # Use log scale if parameter varies by orders of magnitude
    param_values = grouped['param'].values
    if len(param_values) > 1 and max(param_values) / max(min(param_values), 0.001) > 10:
        plt.semilogx(grouped['param'], grouped['score'], marker='o', linestyle='-', color='b')
        plt.fill_between(grouped['param'],
                         grouped['score'] - grouped['std'],
                         grouped['score'] + grouped['std'],
                         alpha=0.1, color='b')
        plt.xlabel(f'{param_name} (log scale)', fontsize=12)
    else:
        plt.plot(grouped['param'], grouped['score'], marker='o', linestyle='-', color='b')
        plt.fill_between(grouped['param'],
                         grouped['score'] - grouped['std'],
                         grouped['score'] + grouped['std'],
                         alpha=0.1, color='b')
        plt.xlabel(f'{param_name}', fontsize=12)

    plt.ylabel('CV Average R² Score', fontsize=12)
    plt.title(f'{param_name} vs. R² Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Run optimization pipeline."""

    # 1. Load data
    data_dir = Path("./data/processed")
    game_logs, players, teams, position_defense = load_data(
        data_dir / "game_logs_features.csv",
        data_dir / "processed_players.csv",
        data_dir / "processed_teams.csv",
        data_dir / "team_vs_position_defense.csv",
    )

    # 2. Merge and prepare
    df = merge_data(game_logs, players, teams, position_defense)
    df = fill_season_to_date_with_baseline(df)

    # 3. Prepare features
    X, y, feature_names = prepare_features(df)

    # 4. Define parameter grid
    param_grid = {
        'min_child_weight': [15],
        'reg_alpha': [0.1],
        'reg_lambda': [3],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }

    # 5. Run optimization with rolling window CV
    # Uses seasons 2022-2024 with window_size=1 (train on one season, test on next)
    # Fold 1: Train on [2022] → Test on 2023
    # Fold 2: Train on [2023] → Test on 2024
    results = optimize_hyperparameters(
        df, X, y,
        param_grid,
        start_season=2022,
        end_season=2024,
        window_size=1
    )

    # 6. Plot results
    plots_dir = Path("./plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for param_name in param_grid.keys():
        plot_param_performance(
            results,
            param_name,
            plots_dir / f"optimize_{param_name}.png"
        )

    # 7. Show summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest parameters found:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nBest CV R² score: {results['best_score']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
