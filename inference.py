"""
inference.py
Production inference for fantasy basketball predictions.

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

from training import (
    load_data, merge_data, fill_season_to_date_with_baseline,
    prepare_features, TARGET,
    ROLLING_FEATURES, SEASON_TO_DATE_FEATURES,
)

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
    model_path: str = "./models/fantasy_predictor",
    data_dir: str = "./data/processed",
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
        return pd.DataFrame()

    player_id = player[0]['id']
    logger.info(f"Found player ID: {player_id} for {player_name}")

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

    # Filter for this player's recent games
    player_normalized = player_name.lower().strip()
    player_games = game_logs[
        game_logs['player_normalized'] == player_normalized
    ].sort_values('game_date', ascending=False).head(20)  # Get last 20 games for safety

    if player_games.empty:
        logger.error(f"No recent games found for {player_name} in cached game logs")
        return pd.DataFrame()

    # Get most recent game's features (has rolling averages and season-to-date stats)
    latest_game = player_games.iloc[0].copy()
    logger.info(f"Using features from most recent game: {latest_game.get('game_date', 'Unknown')}")

    # Load baseline and matchup data
    data_dir = Path(data_dir)
    players_df = pd.read_csv(data_dir / "processed_players.csv")
    teams = pd.read_csv(data_dir / "processed_teams.csv")

    # Get current season
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_season = current_year if current_month >= 10 else current_year - 1

    # Get player baseline stats
    player_normalized = player_name.lower().strip()
    player_baseline = players_df[
        (players_df['player_normalized'] == player_normalized) &
        (players_df['season'] == current_season - 1)
    ]

    # Get position info
    if not player_baseline.empty and 'pos' in player_baseline.columns:
        pos = player_baseline['pos'].iloc[0]
        primary_pos = str(pos).split('-')[0].strip()
        pos_map = {'PG': 'PG', 'SG': 'SG', 'SF': 'SF', 'PF': 'PF', 'C': 'C', 'G': 'PG', 'F': 'SF'}
        primary_pos = pos_map.get(primary_pos, 'SF')
    else:
        primary_pos = 'SF'

    # Load position defense data if available
    pos_def_df = None
    if (data_dir / "team_vs_position_defense.csv").exists():
        pos_def_df = pd.read_csv(data_dir / "team_vs_position_defense.csv")

    # Predict for each upcoming game
    predictions = []

    for idx, game_row in upcoming_games_df.iterrows():
        # Extract opponent team abbreviation
        matchup = game_row.get('VISITOR_TEAM_ABBREVIATION') or game_row.get('HOME_TEAM_ABBREVIATION')
        game_date = game_row.get('GAME_DATE', 'Unknown')
        is_home = '@' not in str(game_row.get('MATCHUP', ''))

        # If the game info has VS_TEAM_ABBREVIATION (more common format)
        if 'VS_TEAM_ABBREVIATION' in game_row:
            opponent = game_row['VS_TEAM_ABBREVIATION']
        elif 'VISITOR_TEAM_ABBREVIATION' in game_row:
            opponent = game_row['VISITOR_TEAM_ABBREVIATION']
        elif 'HOME_TEAM_ABBREVIATION' in game_row:
            opponent = game_row['HOME_TEAM_ABBREVIATION']
        else:
            logger.warning(f"Could not determine opponent for game on {game_date}")
            continue

        # Build feature row
        features = {}

        # Copy rolling features from latest game
        for feat in ROLLING_FEATURES:
            if feat in latest_game:
                features[feat] = latest_game[feat]
            else:
                features[feat] = np.nan

        # Copy season-to-date features
        for feat in SEASON_TO_DATE_FEATURES:
            if feat in latest_game:
                features[feat] = latest_game[feat]
            elif not player_baseline.empty:
                # Fallback to baseline
                baseline_col = feat.replace('_season_avg', '_per_game').replace('reb', 'trb')
                if baseline_col in player_baseline.columns:
                    features[feat] = player_baseline[baseline_col].iloc[0]
                else:
                    features[feat] = np.nan
            else:
                features[feat] = np.nan

        # Get opponent stats
        opp_stats = teams[
            (teams['team'] == opponent) &
            (teams['season'] == current_season - 1)
        ]

        if not opp_stats.empty:
            features['opp_drtg'] = opp_stats['team_drtg'].iloc[0]
            features['opp_pace'] = opp_stats['team_pace'].iloc[0]
        else:
            features['opp_drtg'] = np.nan
            features['opp_pace'] = np.nan

        # Get player's team pace
        if 'team' in latest_game:
            team_stats = teams[
                (teams['team'] == latest_game['team']) &
                (teams['season'] == current_season - 1)
            ]
            if not team_stats.empty:
                features['team_pace'] = team_stats['team_pace'].iloc[0]
            else:
                features['team_pace'] = np.nan
        else:
            features['team_pace'] = np.nan

        # Position one-hot encoding
        features['is_pg'] = 1 if primary_pos == 'PG' else 0
        features['is_sg'] = 1 if primary_pos == 'SG' else 0
        features['is_sf'] = 1 if primary_pos == 'SF' else 0
        features['is_pf'] = 1 if primary_pos == 'PF' else 0
        features['is_c'] = 1 if primary_pos == 'C' else 0

        # Position defense
        if pos_def_df is not None:
            pos_def_row = pos_def_df[
                (pos_def_df['team'] == opponent) &
                (pos_def_df['season'] == current_season - 1) &
                (pos_def_df['position'] == primary_pos)
            ]
            if not pos_def_row.empty:
                features['opp_pos_fg_pct'] = pos_def_row['opp_pos_fg_pct'].iloc[0]
                features['opp_pos_fg_diff'] = pos_def_row['opp_pos_fg_diff'].iloc[0]
            else:
                features['opp_pos_fg_pct'] = np.nan
                features['opp_pos_fg_diff'] = np.nan
        else:
            features['opp_pos_fg_pct'] = np.nan
            features['opp_pos_fg_diff'] = np.nan

        # Situational features
        features['is_back_to_back'] = 0  # TODO: Calculate from game dates

        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[feature_names]

        # Make prediction
        prediction = model.predict(X)[0]

        # Store result
        predictions.append({
            'player': player_name,
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
        logger.info(f"\nPredictions for {player_name}:")
        logger.info(f"\n{results_df.to_string(index=False)}")

    return results_df


def predict_next_game(
    player_name: str,
    opponent: str,
    is_home: bool = True,
    model_path: str = "./models/fantasy_predictor",
    data_dir: str = "./data/processed",
) -> dict:
    """
    Predict a player's fantasy points for their next game.

    Args:
        player_name: Player's full name (e.g., "LeBron James")
        opponent: Opponent team abbreviation (e.g., "GSW")
        is_home: Whether the game is at home
        model_path: Path to trained model
        data_dir: Directory with processed data

    Returns:
        dict with prediction and player info
    """
    from data_fetching import fetch_player_recent_games

    logger.info(f"Predicting next game for {player_name} vs {opponent}")

    # Load model
    model, feature_names = load_model(model_path)

    # Fetch recent games to calculate rolling features
    recent_games = fetch_player_recent_games(player_name, n_games=10)

    if recent_games.empty:
        logger.error(f"Could not fetch recent games for {player_name}")
        return {}

    # Get most recent game's features (has rolling averages)
    latest_game = recent_games.iloc[0].copy()

    # Load baseline and matchup data
    data_dir = Path(data_dir)
    players = pd.read_csv(data_dir / "processed_players.csv")
    teams = pd.read_csv(data_dir / "processed_teams.csv")

    # Get current season
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_season = current_year if current_month >= 10 else current_year - 1

    # Get player baseline stats
    player_normalized = player_name.lower().strip()
    player_baseline = players[
        (players['player_normalized'] == player_normalized) &
        (players['season'] == current_season - 1)
    ]

    # Build feature row
    features = {}

    # Copy rolling features from latest game
    for feat in ROLLING_FEATURES:
        if feat in latest_game:
            features[feat] = latest_game[feat]
        else:
            features[feat] = np.nan

    # Copy season-to-date features
    for feat in SEASON_TO_DATE_FEATURES:
        if feat in latest_game:
            features[feat] = latest_game[feat]
        elif not player_baseline.empty and feat.replace('_season_avg', '_per_game') in player_baseline.columns:
            # Fallback to baseline
            baseline_col = feat.replace('_season_avg', '_per_game').replace('reb', 'trb')
            if baseline_col in player_baseline.columns:
                features[feat] = player_baseline[baseline_col].iloc[0]
            else:
                features[feat] = np.nan
        else:
            features[feat] = np.nan

    # Get opponent stats
    opp_stats = teams[
        (teams['team'] == opponent) &
        (teams['season'] == current_season - 1)
    ]

    if not opp_stats.empty:
        features['opp_drtg'] = opp_stats['team_drtg'].iloc[0]
        features['opp_pace'] = opp_stats['team_pace'].iloc[0]
    else:
        features['opp_drtg'] = np.nan
        features['opp_pace'] = np.nan

    # Get player's team pace
    if 'team' in latest_game:
        team_stats = teams[
            (teams['team'] == latest_game['team']) &
            (teams['season'] == current_season - 1)
        ]
        if not team_stats.empty:
            features['team_pace'] = team_stats['team_pace'].iloc[0]
        else:
            features['team_pace'] = np.nan
    else:
        features['team_pace'] = np.nan

    # Position defense (need position)
    if not player_baseline.empty and 'pos' in player_baseline.columns:
        pos = player_baseline['pos'].iloc[0]
        primary_pos = str(pos).split('-')[0].strip()
        pos_map = {'PG': 'PG', 'SG': 'SG', 'SF': 'SF', 'PF': 'PF', 'C': 'C', 'G': 'PG', 'F': 'SF'}
        primary_pos = pos_map.get(primary_pos, 'SF')

        # Position one-hot
        features['is_pg'] = 1 if primary_pos == 'PG' else 0
        features['is_sg'] = 1 if primary_pos == 'SG' else 0
        features['is_sf'] = 1 if primary_pos == 'SF' else 0
        features['is_pf'] = 1 if primary_pos == 'PF' else 0
        features['is_c'] = 1 if primary_pos == 'C' else 0

        # Position defense
        if (data_dir / "team_vs_position_defense.csv").exists():
            pos_def = pd.read_csv(data_dir / "team_vs_position_defense.csv")
            pos_def_row = pos_def[
                (pos_def['team'] == opponent) &
                (pos_def['season'] == current_season - 1) &
                (pos_def['position'] == primary_pos)
            ]
            if not pos_def_row.empty:
                features['opp_pos_fg_pct'] = pos_def_row['opp_pos_fg_pct'].iloc[0]
                features['opp_pos_fg_diff'] = pos_def_row['opp_pos_fg_diff'].iloc[0]
            else:
                features['opp_pos_fg_pct'] = np.nan
                features['opp_pos_fg_diff'] = np.nan
        else:
            features['opp_pos_fg_pct'] = np.nan
            features['opp_pos_fg_diff'] = np.nan
    else:
        # No position info
        features['is_pg'] = 0
        features['is_sg'] = 0
        features['is_sf'] = 0
        features['is_pf'] = 0
        features['is_c'] = 0
        features['opp_pos_fg_pct'] = np.nan
        features['opp_pos_fg_diff'] = np.nan

    # Situational
    features['is_back_to_back'] = 0  # Assume not back-to-back unless specified

    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])[feature_names]

    # Make prediction
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
    data_dir: str = "./data/processed",
    model_path: str = "./models/fantasy_predictor",
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
    output_path: str = "./data/processed/game_logs_features.csv",
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
    from data_fetching import fetch_and_process
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
    player = "shai gilgeous-alexander"

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
