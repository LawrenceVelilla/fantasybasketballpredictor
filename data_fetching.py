"""
data_fetcher.py
Fetches game logs from nba_api and calculates rolling features.
Output: game_logs_features.parquet (Rows = Individual Games)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging
import time
from nba_api.stats.endpoints import leaguegamelog, playergamelog
from nba_api.stats.static import players, teams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FANTASY_SETTINGS = {
    "pts": 1.0,
    "reb": 1,
    "ast": 1.5,
    "stl": 2.5,
    "blk": 2.5,
    "tov": -1.0,
    "fg_made": 1.0,      
    "fg_missed": -1.0,  
    "ft_made": 0.5,
    "ft_missed": -0.5,
    "three_made": 1.0,  
}

# Type conversion utilities
def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert column to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors='coerce')


def safe_convert_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """Safely convert columns to specified types."""
    df = df.copy()
    
    for col, dtype in type_map.items():
        if col not in df.columns:
            continue
            
        if dtype == 'numeric':
            df[col] = clean_numeric_column(df[col])
        elif dtype == 'int':
            df[col] = clean_numeric_column(df[col]).fillna(0).astype(int)
        elif dtype == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df



def calculate_game_fantasy_points(df: pd.DataFrame) -> pd.Series:
    """
    Calculate actual fantasy points for a single game.
    
    Expects columns: PTS, REB, AST, STL, BLK, TOV
    """
    fpts = pd.Series(0.0, index=df.index)
    
    col_variants = {
        'pts': ['pts', 'pts_per_game', 'PTS'],
        'reb': ['trb', 'trb_per_game', 'reb', 'reb_per_game', 'REB'],
        'ast': ['ast', 'ast_per_game', 'AST'],
        'stl': ['stl', 'stl_per_game', 'STL'],
        'blk': ['blk', 'blk_per_game', 'BLK'],
        'tov': ['tov', 'tov_per_game', 'TOV'],
    }
    
    for stat, multiplier in FANTASY_SETTINGS.items():
        for col_variant in col_variants.get(stat, []):
            if col_variant in df.columns:
                fpts += df[col_variant].fillna(0) * multiplier
                break
    
    return fpts

# Api fetching functions
def fetch_league_game_logs(
    season: str,
    season_type: str = "Regular Season",
    rate_limit_delay: float = 1.5
) -> pd.DataFrame:
    """
    Fetch all player game logs for a given season.
    
    Args:
        season: Season string, e.g., "2023-24"
        season_type: "Regular Season" or "Playoffs"
        rate_limit_delay: Seconds to wait between API calls
    
    Returns:
        DataFrame with all game logs
    """
    logger.info(f"Fetching game logs for {season} ({season_type})")
    
    time.sleep(rate_limit_delay)  
    
    try:
        game_log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",  # Player logs
        )
        df = game_log.get_data_frames()[0]
        logger.info(f"Fetched {len(df)} game logs for {season}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching {season}: {e}")
        return pd.DataFrame()


def fetch_multiple_seasons(
    start_year: int = 2022,
    end_year: int = 2024,
    rate_limit_delay: float = 1.5
) -> pd.DataFrame:
    """
    Fetch game logs for multiple seasons.
    
    Args:
        start_year: Starting season year (e.g., 2018 for 2018-19)
        end_year: Ending season year
        rate_limit_delay: Seconds between API calls
    """
    all_logs = []
    
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"  # e.g., "2023-24"
        
        df = fetch_league_game_logs(season, rate_limit_delay=rate_limit_delay)
        
        if not df.empty:
            df['SEASON'] = year  # Store as integer for easier joins
            df['SEASON_STR'] = season
            all_logs.append(df)
    
    if not all_logs:
        logger.warning("No game logs fetched!")
        return pd.DataFrame()
    
    combined = pd.concat(all_logs, ignore_index=True)
    logger.info(f"Total game logs fetched: {len(combined)}")
    
    return combined

def calculate_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 5, 10],
    stats: List[str] = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT']
) -> pd.DataFrame:
    """
    Calculate rolling averages for each player.
    
    IMPORTANT: Uses shift(1) to avoid data leakage - rolling stats
    are calculated from PREVIOUS games, not including current game.
    """
    df = df.copy()
    
    # Ensure sorted by player and date
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    
    for window in windows:
        logger.info(f"Calculating rolling {window}-game averages")
        
        for stat in stats:
            if stat not in df.columns:
                logger.warning(f"Column {stat} not found, skipping")
                continue
            
            col_name = f"{stat.lower()}_last_{window}"
            
            # shift(1) prevents data leakage
            df[col_name] = (
                df.groupby('PLAYER_ID')[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
    
    return df


def calculate_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days of rest between games for each player.
    """
    df = df.copy()
    
    # Ensure GAME_DATE is datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    
    # Sort by player and date
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    
    # Calculate days since previous game
    df['prev_game_date'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    df['rest_days'] = (df['GAME_DATE'] - df['prev_game_date']).dt.days
    
    # First game of season: default to 3 days rest (reasonable assumption)
    df['rest_days'] = df['rest_days'].fillna(3)
    
    # Cap at reasonable max (e.g., 14 days for injury returns)
    df['rest_days'] = df['rest_days'].clip(upper=14)
    
    # Drop helper column
    df = df.drop(columns=['prev_game_date'])
    
    return df


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse matchup string to extract opponent and home/away.
    
    MATCHUP format: "LAL vs. GSW" (home) or "LAL @ GSW" (away)
    """
    df = df.copy()
    
    if 'MATCHUP' not in df.columns:
        logger.warning("MATCHUP column not found")
        return df
    
    # Home game indicator
    df['is_home'] = df['MATCHUP'].str.contains('vs.', case=False, na=False).astype(int)
    
    # Extract opponent team abbreviation
    df['opponent'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*(\w+)', expand=False)
    
    return df


def add_game_number(df: pd.DataFrame) -> pd.DataFrame:
    """Add season game number for each player (useful for fatigue modeling)."""
    df = df.copy()
    
    df = df.sort_values(['PLAYER_ID', 'SEASON', 'GAME_DATE']).reset_index(drop=True)
    
    df['game_number'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount() + 1
    
    return df


def process_game_logs(
    df: pd.DataFrame,
    rolling_windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Main processing pipeline for game logs.
    
    1. Clean and convert types
    2. Calculate actual fantasy points (target variable)
    3. Calculate rolling features
    4. Calculate rest days
    5. Add matchup features
    6. Select only needed columns
    """
    logger.info("Processing game logs...")
    
    # Type conversions
    type_map = {
        'PTS': 'numeric',
        'REB': 'numeric',
        'AST': 'numeric',
        'STL': 'numeric',
        'BLK': 'numeric',
        'TOV': 'numeric',
        'MIN': 'numeric',
        'FGM': 'numeric',
        'FGA': 'numeric',
        'FG_PCT': 'numeric',
        'FG3M': 'numeric',
        'FG3A': 'numeric',
        'FG3_PCT': 'numeric',
        'FTM': 'numeric',
        'FTA': 'numeric',
        'FT_PCT': 'numeric',
        'OREB': 'numeric',
        'DREB': 'numeric',
        'PF': 'numeric',
        'PLUS_MINUS': 'numeric',
        'GAME_DATE': 'datetime',
        'PLAYER_ID': 'int',
        'GAME_ID': 'string',
        'SEASON': 'int',
    }
    
    df = safe_convert_types(df, type_map)
    
    # Calculate actual fantasy points (TARGET VARIABLE)
    logger.info("Calculating actual fantasy points")
    df['actual_fantasy_pts'] = calculate_game_fantasy_points(df)
    
    # Rolling features (with lag to prevent leakage)
    rolling_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT']
    df = calculate_rolling_features(df, windows=rolling_windows, stats=rolling_stats)
    
    # Rolling fantasy points
    for window in rolling_windows:
        col_name = f"fppg_last_{window}"
        df[col_name] = (
            df.groupby('PLAYER_ID')['actual_fantasy_pts']
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
    
    # Rest days
    df = calculate_rest_days(df)
    
    # Matchup features
    df = add_matchup_features(df)
    
    # Game number
    df = add_game_number(df)
    
    # Normalize player name for joining with processed_players.csv
    if 'PLAYER_NAME' in df.columns:
        df['player_normalized'] = df['PLAYER_NAME'].str.lower().str.strip()
    
    
    # Identifiers (for joining)
    id_cols = {
        'PLAYER_ID': 'player_id',
        'PLAYER_NAME': 'player_name',
        'player_normalized': 'player_normalized',
        'SEASON': 'season',
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'TEAM_ABBREVIATION': 'team',
        'opponent': 'opponent',
    }
    
    # Target variable
    target_cols = {
        'actual_fantasy_pts': 'actual_fantasy_pts',
    }
    
    # Game stats (for reference, not model features)
    game_stat_cols = {
        'PTS': 'pts',
        'REB': 'reb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TOV': 'tov',
        'MIN': 'min',
        'FG_PCT': 'fg_pct',
    }
    
    # Rolling features (model features) - already lowercase from calculate_rolling_features
    rolling_cols = {}
    for window in rolling_windows:
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'min', 'fg_pct']:
            col_name = f"{stat}_last_{window}"
            rolling_cols[col_name] = col_name  # already correct name
        rolling_cols[f"fppg_last_{window}"] = f"fppg_last_{window}"
    
    # Situational features
    situation_cols = {
        'is_home': 'is_home',
        'rest_days': 'rest_days',
        'game_number': 'game_number',
    }
    
    # Combine all column mappings
    all_cols = {**id_cols, **target_cols, **game_stat_cols, **rolling_cols, **situation_cols}
    
    # Keep only columns that exist and rename
    keep_cols = {k: v for k, v in all_cols.items() if k in df.columns}
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)
    
    logger.info(f"Processed {len(df)} game logs")
    logger.info(f"Final columns: {list(df.columns)}")
    
    return df


def fetch_and_process(
    start_year: int = 2022,
    end_year: int = 2024,
    output_path: str = "./data/processed/game_logs_features.csv",
    rate_limit_delay: float = 1.5,
    rolling_windows: List[int] = [3, 5, 10]
) -> pd.DataFrame:
    """
    Full pipeline: fetch from API, process, and save.
    """
    # Fetch
    df = fetch_multiple_seasons(
        start_year=start_year,
        end_year=end_year,
        rate_limit_delay=rate_limit_delay
    )
    
    if df.empty:
        logger.error("No data fetched, exiting")
        return pd.DataFrame()
    
    # Process
    df = process_game_logs(df, rolling_windows=rolling_windows)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving to {output_path}")    
    df.to_csv(output_path, index=False)
    
    return df


def fetch_player_recent_games(
    player_name: str,
    n_games: int = 10,
    rate_limit_delay: float = 0.6
) -> pd.DataFrame:
    """
    Fetch recent games for a specific player (for live inference).
    
    Args:
        player_name: Full player name (e.g., "LeBron James")
        n_games: Number of recent games to fetch
        rate_limit_delay: Seconds to wait before API call
    
    Returns:
        DataFrame with recent game logs
    """
    # Find player ID
    player_list = players.find_players_by_full_name(player_name)
    
    if not player_list:
        logger.error(f"Player not found: {player_name}")
        return pd.DataFrame()
    
    player_id = player_list[0]['id']
    logger.info(f"Found {player_name} with ID {player_id}")
    
    time.sleep(rate_limit_delay)
    
    # Get current season
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # NBA season starts in October
    if current_month >= 10:
        season = f"{current_year}-{str(current_year + 1)[-2:]}"
    else:
        season = f"{current_year - 1}-{str(current_year)[-2:]}"
    
    try:
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        df = game_log.get_data_frames()[0]
        
        # Take most recent n games
        df = df.head(n_games)
        
        # Process features
        df = process_game_logs(df, rolling_windows=[5])
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching games for {player_name}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and process NBA game logs")
    parser.add_argument("--start-year", type=int, default=2022,
                        help="Starting season year")
    parser.add_argument("--end-year", type=int, default=2024,
                        help="Ending season year")
    parser.add_argument("--output", type=str, 
                        default="./data/processed/game_logs_features.csv",
                        help="Output path (.csv or .parquet)")
    parser.add_argument("--delay", type=float, default=0.6,
                        help="Rate limit delay in seconds")
    
    args = parser.parse_args()
    
    df = fetch_and_process(
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output,
        rate_limit_delay=args.delay,
    )
    
    if not df.empty:
        print(f"\nProcessed {len(df)} game logs")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        sample_cols = ['player_name', 'game_date', 'pts', 'actual_fantasy_pts', 
                       'pts_last_5', 'fppg_last_5', 'rest_days', 'is_home', 'opponent']
        sample_cols = [c for c in sample_cols if c in df.columns]
        print(df[sample_cols].head(10))
        print(f"\nData types:\n{df.dtypes}")