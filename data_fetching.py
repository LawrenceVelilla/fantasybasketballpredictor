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
        logger.info(f"Columns: {list(df.columns)}")
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

def fetch_team_game_logs(
    season: str,
    season_type: str = "Regular Season",
    rate_limit_delay: float = 1.5
) -> pd.DataFrame:
    """
    Fetch all team game logs for a given season.

    Args:
        season: Season string, e.g., "2023-24"
        season_type: "Regular Season" or "Playoffs"
        rate_limit_delay: Seconds to wait between API calls

    Returns:
        DataFrame with team game logs
    """
    logger.info(f"Fetching team game logs for {season} ({season_type})")

    time.sleep(rate_limit_delay)

    try:
        game_log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T",  # Team logs
        )
        df = game_log.get_data_frames()[0]
        logger.info(f"Fetched {len(df)} team game logs for {season}")
        return df

    except Exception as e:
        logger.error(f"Error fetching team logs for {season}: {e}")
        return pd.DataFrame()


def fetch_team_stats_multiple_seasons(
    start_year: int = 2022,
    end_year: int = 2024,
    rate_limit_delay: float = 1.5
) -> pd.DataFrame:
    """
    Fetch team game logs for multiple seasons.

    Args:
        start_year: Starting season year (e.g., 2022 for 2022-23)
        end_year: Ending season year
        rate_limit_delay: Seconds between API calls

    Returns:
        DataFrame with team stats including TEAM_FGA, TEAM_FTA, TEAM_TOV
    """
    all_team_logs = []

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"

        df = fetch_team_game_logs(season, rate_limit_delay=rate_limit_delay)

        if not df.empty:
            df['SEASON'] = year
            all_team_logs.append(df)

    if not all_team_logs:
        logger.warning("No team game logs fetched!")
        return pd.DataFrame()

    combined = pd.concat(all_team_logs, ignore_index=True)
    logger.info(f"Total team game logs fetched: {len(combined)}")

    return combined


def merge_team_stats(player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team stats into player game logs for usage rate calculation.

    Joins on GAME_ID and TEAM_ID to get the team's totals for each game.

    Args:
        player_df: Player game logs DataFrame
        team_df: Team game logs DataFrame

    Returns:
        Player DataFrame with team stats columns added
    """
    if team_df.empty:
        logger.warning("Team stats DataFrame is empty, skipping merge")
        return player_df

    # Select only the columns we need from team stats
    team_cols = ['GAME_ID', 'TEAM_ID', 'FGA', 'FTA', 'TOV', 'MIN']
    available_cols = [col for col in team_cols if col in team_df.columns]

    if 'GAME_ID' not in available_cols or 'TEAM_ID' not in available_cols:
        logger.error("Missing GAME_ID or TEAM_ID in team stats")
        return player_df

    team_stats = team_df[available_cols].copy()

    # Rename columns to avoid conflicts and indicate they're team stats
    rename_map = {
        'FGA': 'TEAM_FGA',
        'FTA': 'TEAM_FTA',
        'TOV': 'TEAM_TOV',
        'MIN': 'TEAM_MIN'
    }
    team_stats = team_stats.rename(columns=rename_map)

    # Merge on GAME_ID and TEAM_ID
    merged = player_df.merge(
        team_stats,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )

    logger.info(f"Merged team stats: {len(merged)} rows")

    return merged


def calculate_usage_rate(df: pd.DataFrame) -> pd.Series:
    """
    Calculate usage rate for each player in each game.

    Usage Rate formula from google:
    100 * ((FGA + 0.44 * FTA + TOV) * (Team MIN / 5)) /
          (Player MIN * (Team FGA + 0.44 * Team FTA + Team TOV))

    This estimates the percentage of team possessions a player uses while on court.

    Requires columns: FGA, FTA, TOV, MIN, TEAM_FGA, TEAM_FTA, TEAM_TOV

    Returns:
        Series with usage rate as a percentage (0-100 scale)
    """
    required_cols = ['FGA', 'FTA', 'TOV', 'MIN', 'TEAM_FGA', 'TEAM_FTA', 'TEAM_TOV']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing columns for usage rate: {missing_cols}")
        return pd.Series(np.nan, index=df.index)

    logger.info("Calculating usage rate for each player")

    # Calculate player possessions used
    player_possessions = df['FGA'] + 0.44 * df['FTA'] + df['TOV']

    # Calculate team possessions (approximately)
    team_possessions = df['TEAM_FGA'] + 0.44 * df['TEAM_FTA'] + df['TEAM_TOV']

    # Team minutes divided by 5 (5 players on court)
    team_min_per_player = df.get('TEAM_MIN', 240) / 5

    # Usage rate formula
    denominator = df['MIN'] * team_possessions
    usg = np.where(
        denominator > 0,
        100 * (player_possessions * team_min_per_player) / denominator,
        np.nan
    )

    return pd.Series(usg, index=df.index)


def calculate_season_to_date_features(
    df: pd.DataFrame,
    stats: List[str] = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT']
) -> pd.DataFrame:
    """
    Calculate season-to-date averages for each player.
    
    Uses expanding window with shift(1) to avoid data leakage 
    Averages are calculated from all PREVIOUS games in the season,
    not including the current game.

    For early season games with few prior games, these values will be based
    on a small sample. The model should use baseline features as fallback.
    """
    df = df.copy()

    # Ensure sorted by player, season, and date
    df = df.sort_values(['PLAYER_ID', 'SEASON', 'GAME_DATE']).reset_index(drop=True)

    logger.info("Calculating season-to-date averages")

    for stat in stats:
        if stat not in df.columns:
            logger.warning(f"Column {stat} not found, skipping")
            continue

        col_name = f"{stat.lower()}_season_avg"

        # shift(1) prevents data leakage - only uses games before current
        # expanding() calculates cumulative average from start of season
        df[col_name] = (
            df.groupby(['PLAYER_ID', 'SEASON'])[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

    # Also calculate season-to-date fantasy points
    df['fppg_season_avg'] = (
        df.groupby(['PLAYER_ID', 'SEASON'])['actual_fantasy_pts']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Add game count in season (useful for knowing sample size)
    df['games_played_season'] = (
        df.groupby(['PLAYER_ID', 'SEASON']).cumcount()  # 0-indexed count of games before this one
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

    # Season-to-date features
    season_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT']
    df = calculate_season_to_date_features(df, stats=season_stats)

    # Rest days
    df = calculate_rest_days(df)
    
    # Matchup features
    df = add_matchup_features(df)
    
    # Game number
    df = add_game_number(df)

    # Usage rate 
    # Merge team stats first
    if 'TEAM_FGA' in df.columns:
        df['usage_rate'] = calculate_usage_rate(df)

        # Calculate rolling usage rate (last 3 and last 5 games)
        logger.info("Calculating rolling usage rate averages")
        for window in [3, 5]:
            col_name = f"usage_rate_last_{window}"
            df[col_name] = (
                df.groupby('PLAYER_ID')['usage_rate']
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
    else:
        logger.info("Team stats not available, skipping usage rate calculation")

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

    # Rolling usage rate features
    for window in [3, 5]:
        col_name = f"usage_rate_last_{window}"
        rolling_cols[col_name] = col_name

    # Season-to-date features (model features)
    season_avg_cols = {}
    for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'min', 'fg_pct']:
        col_name = f"{stat}_season_avg"
        season_avg_cols[col_name] = col_name
    season_avg_cols['fppg_season_avg'] = 'fppg_season_avg'
    season_avg_cols['games_played_season'] = 'games_played_season'

    # Situational features
    situation_cols = {
        'is_home': 'is_home',
        'rest_days': 'rest_days',
        'game_number': 'game_number',
        'usage_rate': 'usage_rate',
    }
    
    # Combine all column mappings
    all_cols = {**id_cols, **target_cols, **game_stat_cols, **rolling_cols, **season_avg_cols, **situation_cols}
    
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
    rolling_windows: List[int] = [3, 5, 10],
    include_team_stats: bool = True
) -> pd.DataFrame:
    """
    Full pipeline: fetch from API, process, and save.

    Args:
        start_year: Starting season year
        end_year: Ending season year
        output_path: Path to save processed data
        rate_limit_delay: Seconds between API calls
        rolling_windows: Windows for rolling averages
        include_team_stats: Whether to fetch and merge team stats for usage rate
    """
    # Fetch player game logs
    df = fetch_multiple_seasons(
        start_year=start_year,
        end_year=end_year,
        rate_limit_delay=rate_limit_delay
    )

    if df.empty:
        logger.error("No data fetched, exiting")
        return pd.DataFrame()

    # Fetch and merge team stats if requested
    if include_team_stats:
        logger.info("Fetching team stats for usage rate calculation...")
        team_df = fetch_team_stats_multiple_seasons(
            start_year=start_year,
            end_year=end_year,
            rate_limit_delay=rate_limit_delay
        )
        df = merge_team_stats(df, team_df)

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

        # Add PLAYER_ID if missing (PlayerGameLog might not include it)
        if 'PLAYER_ID' not in df.columns:
            df['PLAYER_ID'] = player_id

        # Add SEASON if missing
        if 'SEASON' not in df.columns:
            # Extract year from season string (e.g., "2024-25" -> 2024)
            season_year = int(season.split('-')[0])
            df['SEASON'] = season_year

        logger.info("Processing game logs...")
        # Process features
        df = process_game_logs(df, rolling_windows=[3, 5])

        return df

    except Exception as e:
        logger.error(f"Error fetching games for {player_name}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch and process NBA game logs")
    parser.add_argument("--start-year", type=int, default=2020,
                        help="Starting season year")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="Ending season year")
    parser.add_argument("--output", type=str, 
                        default="./data/processed/game_logs_features.csv",
                        help="Output path (.csv or .parquet)")
    parser.add_argument("--delay", type=float, default=1.5,
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