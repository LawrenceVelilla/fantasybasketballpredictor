"""
features.py
Shared feature engineering utilities for training and inference.

This module contains common functions used by both training.py and inference.py
to ensure consistency in feature building.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def get_current_season() -> int:
    """Get the current NBA season year."""
    from datetime import datetime
    current_year = datetime.now().year
    current_month = datetime.now().month
    # NBA season starts in October
    return current_year if current_month >= 10 else current_year - 1


def normalize_position(pos: str) -> str:
    """
    Normalize position string to primary position.

    Args:
        pos: Position string (e.g., "PG", "SG-SF", "G-F")

    Returns:
        Primary position: PG, SG, SF, PF, or C
    """
    if pd.isna(pos):
        return 'SF'  # Default fallback

    # Take first position if multi-position
    primary = str(pos).split('-')[0].strip()

    # Map common variations
    pos_map = {
        'PG': 'PG', 'SG': 'SG', 'SF': 'SF', 'PF': 'PF', 'C': 'C',
        'G': 'PG', 'F': 'SF', 'G-F': 'SG', 'F-G': 'SF', 'F-C': 'PF', 'C-F': 'C'
    }

    return pos_map.get(primary, 'SF')


def encode_position_onehot(position: str) -> Dict[str, int]:
    """
    One-hot encode position.

    Args:
        position: Normalized position (PG, SG, SF, PF, C)

    Returns:
        Dictionary with is_pg, is_sg, is_sf, is_pf, is_c
    """
    return {
        'is_pg': 1 if position == 'PG' else 0,
        'is_sg': 1 if position == 'SG' else 0,
        'is_sf': 1 if position == 'SF' else 0,
        'is_pf': 1 if position == 'PF' else 0,
        'is_c': 1 if position == 'C' else 0,
    }


def get_player_position(
    player_name: str,
    players_df: pd.DataFrame,
    season: Optional[int] = None
) -> str:
    """
    Get player's position from processed_players.csv.

    Args:
        player_name: Player's full name
        players_df: DataFrame from processed_players.csv
        season: Season to look up (default: current season - 1)

    Returns:
        Normalized position string
    """
    if season is None:
        season = get_current_season() - 1

    player_normalized = player_name.lower().strip()
    player_row = players_df[
        (players_df['player_normalized'] == player_normalized) &
        (players_df['season'] == season)
    ]

    if player_row.empty or 'pos' not in players_df.columns:
        logger.warning(f"Position not found for {player_name}, using default SF")
        return 'SF'

    pos = player_row['pos'].iloc[0]
    return normalize_position(pos)


def get_team_stats(
    team: str,
    teams_df: pd.DataFrame,
    season: int,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Get team stats (pace, DRTG, etc.) for a specific season.

    Args:
        team: Team abbreviation (e.g., "LAL")
        teams_df: DataFrame from processed_teams.csv
        season: Season year
        prefix: Prefix for returned keys (e.g., 'opp_' for opponent stats)

    Returns:
        Dictionary with team stats
    """
    team_row = teams_df[
        (teams_df['team'] == team) &
        (teams_df['season'] == season)
    ]

    stats = {}

    if not team_row.empty:
        if 'team_drtg' in teams_df.columns:
            stats[f'{prefix}drtg'] = team_row['team_drtg'].iloc[0]
        if 'team_pace' in teams_df.columns:
            stats[f'{prefix}pace'] = team_row['team_pace'].iloc[0]
        if 'team_ortg' in teams_df.columns:
            stats[f'{prefix}ortg'] = team_row['team_ortg'].iloc[0]
    else:
        # Fill with NaN if team not found
        stats[f'{prefix}drtg'] = np.nan
        stats[f'{prefix}pace'] = np.nan
        if f'{prefix}ortg' in teams_df.columns:
            stats[f'{prefix}ortg'] = np.nan

    return stats


def get_position_defense_stats(
    opponent: str,
    position: str,
    pos_def_df: pd.DataFrame,
    season: int
) -> Dict[str, float]:
    """
    Get position-specific defensive stats for an opponent.

    Args:
        opponent: Opponent team abbreviation
        position: Player's position (PG, SG, SF, PF, C)
        pos_def_df: DataFrame from team_vs_position_defense.csv
        season: Season year

    Returns:
        Dictionary with opp_pos_fg_pct and opp_pos_fg_diff
    """
    stats = {
        'opp_pos_fg_pct': np.nan,
        'opp_pos_fg_diff': np.nan,
    }

    if pos_def_df is None or pos_def_df.empty:
        return stats

    pos_def_row = pos_def_df[
        (pos_def_df['team'] == opponent) &
        (pos_def_df['season'] == season) &
        (pos_def_df['position'] == position)
    ]

    if not pos_def_row.empty:
        if 'opp_pos_fg_pct' in pos_def_df.columns:
            stats['opp_pos_fg_pct'] = pos_def_row['opp_pos_fg_pct'].iloc[0]
        if 'opp_pos_fg_diff' in pos_def_df.columns:
            stats['opp_pos_fg_diff'] = pos_def_row['opp_pos_fg_diff'].iloc[0]

    return stats


def build_prediction_features(
    player_recent_stats: Dict[str, float],
    opponent: str,
    player_team: str,
    position: str,
    teams_df: pd.DataFrame,
    pos_def_df: Optional[pd.DataFrame],
    season: int,
    is_back_to_back: bool = False,
) -> Dict[str, float]:
    """
    Build complete feature dictionary for a single prediction.

    This is the SINGLE SOURCE OF TRUTH for feature building.
    Both training and inference should use this function.

    Args:
        player_recent_stats: Dict with rolling/season-to-date stats from recent games
        opponent: Opponent team abbreviation
        player_team: Player's team abbreviation
        position: Player's normalized position
        teams_df: Teams DataFrame
        pos_def_df: Position defense DataFrame (optional)
        season: Season year
        is_back_to_back: Whether this is a back-to-back game

    Returns:
        Complete feature dictionary ready for model input
    """
    features = {}

    # Copy player's recent stats (rolling and season-to-date features)
    features.update(player_recent_stats)

    # Opponent stats
    opp_stats = get_team_stats(opponent, teams_df, season, prefix='opp_')
    features.update(opp_stats)

    # Player's team stats
    team_stats = get_team_stats(player_team, teams_df, season, prefix='team_')
    features['team_pace'] = team_stats.get('team_pace', np.nan)

    # Position one-hot encoding
    pos_encoding = encode_position_onehot(position)
    features.update(pos_encoding)

    # Position defense
    if pos_def_df is not None:
        pos_def_stats = get_position_defense_stats(opponent, position, pos_def_df, season)
        features.update(pos_def_stats)
    else:
        features['opp_pos_fg_pct'] = np.nan
        features['opp_pos_fg_diff'] = np.nan

    # Situational features
    features['is_back_to_back'] = 1 if is_back_to_back else 0

    return features
