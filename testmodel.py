"""
testmodel.py
Main training pipeline CLI for fantasy basketball predictor.

Orchestrates training, evaluation, and cross-validation using modular components.

Usage:
    python testmodel.py                    # Train with default settings
    python testmodel.py --cv               # Train with cross-validation
    python testmodel.py --no-plots         # Skip plot generation
"""

from pathlib import Path
from typing import Tuple
import logging
from training import load_data, merge_data, fill_season_to_date_with_baseline, prepare_features, split_by_season, train_model, save_model
from evaluation import evaluate_model, get_feature_importance, plot_all, rolling_window_cv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pipeline(
    game_logs_path: str = "./data/processed/game_logs_features.csv",
    players_path: str = "./data/processed/processed_players.csv",
    teams_path: str = "./data/processed/processed_teams.csv",
    position_defense_path: str = "./data/processed/team_vs_position_defense.csv",
    model_output_path: str = "./models/fantasy_predictor",
    plots_output_dir: str = "./plots",
    test_season: int = 2024,
    generate_plots: bool = False,
):
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
    logger.info("Filling season-to-date NaN values with baseline stats")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fantasy basketball predictor")
    parser.add_argument("--data-dir", type=str, default="./data/processed",
                        help="Directory containing processed CSVs")
    parser.add_argument("--model-output", type=str, default="./models/fantasy_predictor",
                        help="Output path for trained model")
    parser.add_argument("--plots-dir", type=str, default="./plots",
                        help="Output directory for plots")
    parser.add_argument("--test-season", type=int, default=2025,
                        help="Season to use as test set")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--cv", action="store_true",
                        help="Run cross-validation")
    parser.add_argument("--cv-type", type=str, default="rolling", choices=["rolling", "expanding"],
                        help="CV type: rolling (handles concept drift, recommended) or expanding (uses all historical data)")
    parser.add_argument("--cv-window", type=int, default=1,
                        help="For rolling CV: number of seasons in training window (default=1)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.cv:
        logger.info(f"Running {args.cv_type} window cross-validation...")

        # if args.cv_type == "rolling":
        cv_results = rolling_window_cv(
                game_logs_path=data_dir / "game_logs_features.csv",
                players_path=data_dir / "processed_players.csv",
                teams_path=data_dir / "processed_teams.csv",
                position_defense_path=data_dir / "team_vs_position_defense.csv",
                start_season=2020,
                end_season=args.test_season,
                window_size=args.cv_window,
            )
        # else:  # expanding
        #     cv_results = expanding_window_cv(
        #         game_logs_path=data_dir / "game_logs_features.csv",
        #         players_path=data_dir / "processed_players.csv",
        #         teams_path=data_dir / "processed_teams.csv",
        #         position_defense_path=data_dir / "team_vs_position_defense.csv",
        #         start_season=2020,
        #         end_season=args.test_season,
        #     )

        print("\n")
        print("CROSS-VALIDATION COMPLETE")
        print(f"\nAverage Performance Across Folds:")
        print(f"  MAE:  {cv_results['avg_metrics']['mae']:.2f} fantasy points")
        print(f"  RMSE: {cv_results['avg_metrics']['rmse']:.2f} fantasy points")
        print(f"  R²:   {cv_results['avg_metrics']['r2']:.3f}")
        print("\nFold-by-Fold Results:")
        for fold in cv_results['fold_metrics']:
            print(f"  Fold {fold['fold']}: Train {fold['train_seasons']} → Test {fold['test_season']}")
            print(f"    MAE: {fold['mae']:.2f}, RMSE: {fold['rmse']:.2f}, R²: {fold['r2']:.3f}")

    # Always run regular training pipeline
    logger.info("\nRunning regular training pipeline...")
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
    print("TRAINING COMPLETE")
    print(f"\nModel saved to: {args.model_output}.joblib")
    if not args.no_plots:
        print(f"Plots saved to: {args.plots_dir}/")
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {metrics['mae']:.2f} fantasy points")
    print(f"  RMSE: {metrics['rmse']:.2f} fantasy points")
    print(f"  R²:   {metrics['r2']:.3f}")
    print(f"\nTop 5 Features:")
    print(importance.head(5).to_string(index=False))