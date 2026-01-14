"""
inference.py
Production inference for fantasy basketball predictions.

python3 -m engine.inference

This module handles:
- Loading trained models
- Predicting next game for players (webapp/API)
- Evaluating player performance on test set
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from nba_api.stats.endpoints import playernextngames
from nba_api.stats.static import players as nba_players


from .training import (
    load_data, merge_data, fill_season_to_date_with_baseline,
    prepare_features, TARGET,
    ROLLING_FEATURES, SEASON_TO_DATE_FEATURES,
)
from .features import (
    get_current_season, get_player_position, build_prediction_features
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "processed"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "fantasy_predictor"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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



def predict_player_next_n_games(
    player_name: str,
    n_games: int = 3,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Predict a player's fantasy points for their next N upcoming games.

    Uses cached game logs for features (efficient, no rate limiting) and
    NBA API only to fetch the upcoming game schedule.

    Args:
        player_name: Player's full name (e.g., "LeBron James")
        n_games: Number of upcoming games to predict (1 or 3)
        model_path: Path to trained model
        data_dir: Directory with processed data (must contain game_logs_features.csv)

    Returns:
        DataFrame with predictions for each upcoming game
    """
    logger.info(f"Predicting next {n_games} games for {player_name}")

    # Get player ID from NBA API
    all_players = nba_players.get_players()
    player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]

    if not player:
        logger.error(f"Player '{player_name}' not found in NBA API")
        # Try partial match
        partial_matches = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
        if partial_matches:
            logger.info(f"Did you mean one of these? {[p['full_name'] for p in partial_matches[:5]]}")
        return pd.DataFrame()

    player_id = player[0]['id']
    actual_player_name = player[0]['full_name']  # Use exact name from API
    logger.info(f"Found player: {actual_player_name} (ID: {player_id})")

    # Fetch next N games schedule from NBA API (only for opponent/date info)
    try:
        next_games = playernextngames.PlayerNextNGames(
            number_of_games=n_games,
            player_id=player_id
        )
        upcoming_games_df = next_games.get_data_frames()[0]

        if upcoming_games_df.empty:
            logger.warning(f"No upcoming games found for {player_name}")
            return pd.DataFrame()

        logger.info(f"Found {len(upcoming_games_df)} upcoming games")

    except Exception as e:
        logger.error(f"Error fetching next games from NBA API: {e}")
        return pd.DataFrame()

    # Load model
    model, feature_names = load_model(model_path)

    # Load cached game logs (with pre-calculated features) for this player
    data_dir = Path(data_dir)
    game_logs_path = data_dir / "game_logs_features.csv"

    if not game_logs_path.exists():
        logger.error(f"Game logs not found at {game_logs_path}. Please run data fetching first.")
        return pd.DataFrame()

    logger.info(f"Loading cached game logs from {game_logs_path}")
    game_logs = pd.read_csv(game_logs_path)

    # Filter for this player's recent games (use actual name from API)
    player_normalized = actual_player_name.lower().strip()
    player_games = game_logs[
        game_logs['player_normalized'] == player_normalized
    ].sort_values('game_date', ascending=False).head(20)  # Get last 20 games for safety

    if player_games.empty:
        logger.error(f"No recent games found for {actual_player_name} in cached game logs")
        logger.info(f"Available players in cache: {game_logs['player_normalized'].unique()[:10]}")
        return pd.DataFrame()

    # Verify we got the right player
    found_player_name = player_games.iloc[0].get('player_name', 'Unknown')
    logger.info(f"Found {len(player_games)} recent games for {found_player_name}")

    # Get most recent game's features (has rolling averages and season-to-date stats)
    latest_game = player_games.iloc[0].copy()
    logger.info(f"Using features from most recent game: {latest_game.get('game_date', 'Unknown')}")

    # Extract player's recent stats (rolling + season-to-date)
    player_recent_stats = {}
    for feat in ROLLING_FEATURES + SEASON_TO_DATE_FEATURES:
        player_recent_stats[feat] = latest_game.get(feat, np.nan)

    # Get player's team from latest game
    player_team = latest_game.get('team', None)
    if not player_team:
        logger.error("Could not determine player's team from recent games")
        return pd.DataFrame()

    # Load matchup data
    players_df = pd.read_csv(data_dir / "processed_players.csv")
    teams_df = pd.read_csv(data_dir / "processed_teams.csv")

    # Load position defense data if available
    pos_def_df = None
    if (data_dir / "team_vs_position_defense.csv").exists():
        pos_def_df = pd.read_csv(data_dir / "team_vs_position_defense.csv")

    # Get current season and player position
    current_season = get_current_season()
    player_position = get_player_position(actual_player_name, players_df, current_season - 1)
    logger.info(f"Player position: {player_position}, Team: {player_team}")

    # Predict for each upcoming game
    predictions = []

    for _, game_row in upcoming_games_df.iterrows():
        game_date = game_row.get('GAME_DATE', 'Unknown')
        home_team = game_row.get('HOME_TEAM_ABBREVIATION', '')
        visitor_team = game_row.get('VISITOR_TEAM_ABBREVIATION', '')

        # Determine if player's team is home or away, and who the opponent is
        if player_team == home_team:
            # Player is home team
            is_home = True
            opponent = visitor_team
        elif player_team == visitor_team:
            # Player is away team
            is_home = False
            opponent = home_team
        else:
            logger.warning(f"Player team '{player_team}' doesn't match home '{home_team}' or visitor '{visitor_team}'")
            continue

        if not opponent:
            logger.warning(f"Could not determine opponent for game on {game_date}")
            continue

        logger.info(f"  {game_date}: {player_team} {'vs' if is_home else '@'} {opponent}")

        # Build features using shared utility function
        features = build_prediction_features(
            player_recent_stats=player_recent_stats,
            opponent=opponent,
            player_team=player_team,
            position=player_position,
            teams_df=teams_df,
            pos_def_df=pos_def_df,
            season=current_season - 1,  # Use last season's team/matchup data
            is_back_to_back=False,  # TODO: Calculate from game dates
        )

        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[feature_names]

        # Make prediction
        prediction = model.predict(X)[0]

        # Store result
        predictions.append({
            'player': actual_player_name,
            'game_date': game_date,
            'opponent': opponent,
            'is_home': is_home,
            'predicted_fpts': round(prediction, 1),
            'season_avg_fpts': round(features.get('fppg_season_avg', 0), 1),
            'last_5_avg_fpts': round(features.get('fppg_last_5', 0), 1),
            'last_3_avg_fpts': round(features.get('fppg_last_3', 0), 1),
        })

    results_df = pd.DataFrame(predictions)

    if not results_df.empty:
        logger.info(f"\nPredictions for {actual_player_name}:")
        logger.info(f"\n{results_df.to_string(index=False)}")

    return results_df


def predict_next_game(
    player_name: str,
    opponent: str,
    is_home: bool = True,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict:
    """
    Predict a player's fantasy points for a specific matchup.

    Uses cached game logs for efficiency. Simpler alternative to predict_player_next_n_games
    when you already know the opponent.

    Args:
        player_name: Player's full name (e.g., "LeBron James")
        opponent: Opponent team abbreviation (e.g., "GSW")
        is_home: Whether the game is at home
        model_path: Path to trained model
        data_dir: Directory with processed data

    Returns:
        dict with prediction and player info
    """
    logger.info(f"Predicting next game for {player_name} vs {opponent}")

    # Load model
    model, feature_names = load_model(model_path)

    # Load cached game logs
    data_dir = Path(data_dir)
    game_logs_path = data_dir / "game_logs_features.csv"

    if not game_logs_path.exists():
        logger.error(f"Game logs not found at {game_logs_path}")
        return {}

    game_logs = pd.read_csv(game_logs_path)

    # Get player's recent games
    player_normalized = player_name.lower().strip()
    player_games = game_logs[
        game_logs['player_normalized'] == player_normalized
    ].sort_values('game_date', ascending=False).head(10)

    if player_games.empty:
        logger.error(f"No recent games found for {player_name}")
        return {}

    # Get latest game features
    latest_game = player_games.iloc[0].copy()

    # Extract player's recent stats
    player_recent_stats = {}
    for feat in ROLLING_FEATURES + SEASON_TO_DATE_FEATURES:
        player_recent_stats[feat] = latest_game.get(feat, np.nan)

    # Get player's team
    player_team = latest_game.get('team', None)
    if not player_team:
        logger.error("Could not determine player's team")
        return {}

    # Load matchup data
    players_df = pd.read_csv(data_dir / "processed_players.csv")
    teams_df = pd.read_csv(data_dir / "processed_teams.csv")

    # Load position defense if available
    pos_def_df = None
    if (data_dir / "team_vs_position_defense.csv").exists():
        pos_def_df = pd.read_csv(data_dir / "team_vs_position_defense.csv")

    # Get current season and position
    current_season = get_current_season()
    player_position = get_player_position(player_name, players_df, current_season - 1)

    # Build features using shared utility
    features = build_prediction_features(
        player_recent_stats=player_recent_stats,
        opponent=opponent,
        player_team=player_team,
        position=player_position,
        teams_df=teams_df,
        pos_def_df=pos_def_df,
        season=current_season - 1,
        is_back_to_back=False,
    )

    # Create DataFrame and predict
    X = pd.DataFrame([features])[feature_names]
    prediction = model.predict(X)[0]

    result = {
        'player': player_name,
        'opponent': opponent,
        'is_home': is_home,
        'predicted_fpts': round(prediction, 1),
        'season_avg': round(features.get('fppg_season_avg', 0), 1),
        'last_5_avg': round(features.get('fppg_last_5', 0), 1),
    }

    logger.info(f"Prediction: {result['predicted_fpts']} fpts")

    return result

def evaluate_player(
    player_name: str,
    test_season: int = 2024,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """
    Evaluate model predictions for a specific player on the test set.
    Shows actual vs predicted with detailed metrics.

    Args:
        player_name: Player's name (partial match OK)
        test_season: Season to evaluate
        data_dir: Directory with processed data
        model_path: Path to trained model

    Returns:
        DataFrame with game-by-game predictions and actual values
    """
    logger.info(f"Evaluating {player_name} for {test_season} season")

    # Load model
    model, feature_names = load_model(model_path)

    # Load data
    data_dir = Path(data_dir)
    game_logs, players, teams, position_defense = load_data(
        data_dir / "game_logs_features.csv",
        data_dir / "processed_players.csv",
        data_dir / "processed_teams.csv",
        data_dir / "team_vs_position_defense.csv",
    )

    # Merge data
    df = merge_data(game_logs, players, teams, position_defense)
    df = fill_season_to_date_with_baseline(df)

    # Filter for player and test season
    player_normalized = player_name.lower().strip()
    mask = (
        df['player_normalized'].str.contains(player_normalized, na=False) &
        (df['season'] == test_season)
    )

    if mask.sum() == 0:
        logger.error(f"No games found for '{player_name}' in {test_season}")
        return pd.DataFrame()

    player_df = df[mask].copy()

    # Prepare features and predict
    X = player_df[feature_names].copy()
    y_true = player_df[TARGET].copy()

    player_df['predicted_fpts'] = model.predict(X)
    player_df['error'] = y_true - player_df['predicted_fpts']
    player_df['abs_error'] = player_df['error'].abs()

    # Calculate metrics
    mae = mean_absolute_error(y_true, player_df['predicted_fpts'])
    rmse = np.sqrt(mean_squared_error(y_true, player_df['predicted_fpts']))
    r2 = r2_score(y_true, player_df['predicted_fpts'])

    # Display
    print(f"\n{'='*80}")
    print(f"PLAYER EVALUATION: {player_name.upper()} ({test_season})")
    print(f"{'='*80}\n")

    display_cols = ['game_date', 'opponent', 'is_home', TARGET, 'predicted_fpts', 'error', 'abs_error']
    display_cols = [c for c in display_cols if c in player_df.columns]

    display_df = player_df[display_cols].copy()
    display_df['predicted_fpts'] = display_df['predicted_fpts'].round(2)
    display_df['error'] = display_df['error'].round(1)
    display_df['abs_error'] = display_df['abs_error'].round(1)

    # Rename for display
    display_df = display_df.rename(columns={
        TARGET: 'actual_fpts',
        'game_date': 'date',
    })

    print(display_df.to_string(index=False))

    # Summary
    print(f"\n")
    print(f"SUMMARY STATISTICS")
    print(f"  Games Played:     {len(player_df)}")
    print(f"  Actual Avg:       {y_true.mean():.1f} fpts")
    print(f"  Predicted Avg:    {player_df['predicted_fpts'].mean():.1f} fpts")
    print(f"  MAE:              {mae:.2f} fpts")
    print(f"  RMSE:             {rmse:.2f} fpts")
    print(f"  R²:               {r2:.3f}")
    print(f"  Bias:             {player_df['error'].mean():+.1f} (positive = underpredicting)\n")

    return display_df


def update_game_logs_cache(
    output_path: str | Path = DEFAULT_DATA_DIR / "game_logs_features.csv",
    seasons_to_fetch: int = 2,
) -> None:
    """
    Update the cached game logs with latest data from NBA API.

    This should be run once per day (e.g., via cron job or scheduled task)
    to keep the cache fresh for inference.

    Args:
        output_path: Where to save the updated game logs
        seasons_to_fetch: How many recent seasons to fetch (default: 2 for current + last season)
    """
    from .data_fetching import fetch_and_process
    from datetime import datetime

    logger.info("Updating game logs cache...")

    # Get current season
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_season = current_year if current_month >= 10 else current_year - 1

    # Fetch recent seasons
    start_year = current_season - (seasons_to_fetch - 1)
    end_year = current_season

    logger.info(f"Fetching seasons {start_year} to {end_year}")

    df = fetch_and_process(
        start_year=start_year,
        end_year=end_year,
        output_path=output_path,
        rate_limit_delay=1.5,
    )

    if not df.empty:
        logger.info(f"Cache updated successfully: {len(df)} game logs saved to {output_path}")
    else:
        logger.error("Failed to update cache")


if __name__ == "__main__":
    """Example usage of prediction functions."""

    # Example: Predict next 3 games for a player
    player = "keyonte george"

    print("=" * 80)
    print(f"PREDICTING NEXT 3 GAMES FOR {player.upper()}")
    print("=" * 80)

    predictions = predict_player_next_n_games(
        player_name=player,
        n_games=3,
    )

    if not predictions.empty:
        print("\nPREDICTIONS SUMMARY")
        print(predictions.to_string(index=False))
    else:
        print(f"\nNo predictions available for {player}")
