"""
data_processor.py
Processes Kaggle NBA datasets to create static player season averages.
Output: processed_players.parquet (Rows = Player-Seasons)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Scoring settings for fantasy points calculation
# Can be adjusted as needed
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


def clean_string_column(series: pd.Series) -> pd.Series:
    """Normalize string columns: strip whitespace, handle NaN."""
    return series.astype(str).str.strip().replace('nan', np.nan)


def convert_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Safely convert columns to specified types.
    type_map: {'column_name': 'numeric' | 'string' | 'int'}
    """
    df = df.copy()
    
    for col, dtype in type_map.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping.")
            continue
            
        if dtype == 'numeric':
            df[col] = clean_numeric_column(df[col])
        elif dtype == 'string':
            df[col] = clean_string_column(df[col])
        elif dtype == 'int':
            df[col] = clean_numeric_column(df[col])
            df[col] = df[col].fillna(0).astype(int)
    
    return df


# Points calculation
def calculate_fantasy_points(
    df: pd.DataFrame,
    settings: dict = FANTASY_SETTINGS,
) -> pd.Series:
    """
    Calculate fantasy points based on scoring settings.
    
    Handles multiple column naming conventions:
    - pts, trb, ast, stl, blk, tov (standard)
    - pts_per_game, trb_per_game, etc. (Kaggle format)
    - PTS, REB, AST, etc. (NBA API format)
    
    Returns: Series of fantasy point values
    """
    # Map stat names to possible column names (in priority order)
    col_variants = {
        'pts': ['pts', 'pts_per_game', 'PTS'],
        'reb': ['trb', 'trb_per_game', 'reb', 'reb_per_game', 'REB'],
        'ast': ['ast', 'ast_per_game', 'AST'],
        'stl': ['stl', 'stl_per_game', 'STL'],
        'blk': ['blk', 'blk_per_game', 'BLK'],
        'tov': ['tov', 'tov_per_game', 'TOV'],
    }
    
    fpts = pd.Series(0.0, index=df.index)
    
    for stat, multiplier in settings.items():
        if stat not in col_variants:
            continue
            
        # Find the first matching column
        for col in col_variants[stat]:
            if col in df.columns:
                fpts += df[col].fillna(0) * multiplier
                break
    
    return fpts

# Loading functions
def load_player_per_game(filepath: str | Path) -> pd.DataFrame:
    """Load and clean Player Per Game stats."""
    logger.info(f"Loading Player Per Game from {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Common names: player, season, tm, g, gs, mp, fg, fga, fg%, 3p, 3pa, 3p%,
    #               ft, fta, ft%, orb, drb, trb, ast, stl, blk, tov, pf, pts
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Type conversions 
    type_map = {
        'season': 'int',
        'g': 'int',
        'gs': 'int',
        'pts_per_game': 'numeric',
        'ast_per_game': 'numeric',
        'trb_per_game': 'numeric',
        'stl_per_game': 'numeric',
        'blk_per_game': 'numeric',
        'tov_per_game': 'numeric',
        'mp_per_game': 'numeric',
        'fg_percent': 'numeric',
        'x3p_percent': 'numeric',
        'ft_percent': 'numeric',
        'orb_per_game': 'numeric',
        'drb_per_game': 'numeric',
    }
    
    df = convert_types(df, type_map)
    
    return df


def load_player_advanced(filepath: str | Path) -> pd.DataFrame:
    """Load and clean Player Advanced stats."""
    logger.info(f"Loading Player Advanced from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    type_map = {
        'season': 'int',
        'per': 'numeric',
        'ts_percent': 'numeric',
        'usg_percent': 'numeric',
        'ws': 'numeric',
        'bpm': 'numeric',
        'vorp': 'numeric',
    }
    
    df = convert_types(df, type_map)
    
    return df


def load_team_stats(filepath: str | Path) -> pd.DataFrame:
    """Load and clean Team Stats Per Game."""
    logger.info(f"Loading Team Stats from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    type_map = {
        'season': 'int',
        'pace': 'numeric',
        'ortg': 'numeric',
        'drtg': 'numeric',
    }
    
    df = convert_types(df, type_map)
    
    return df


# Processing pipeliness
def process_players(
    player_per_game_path: str | Path,
    player_advanced_path: str | Path,
    output_path: str | Path,
    min_year: int = 2022,
    min_games: int = 10,
) -> pd.DataFrame:
    """
    Process player season averages.
    
    1. Load Player Per Game and Advanced CSVs
    2. Filter for years >= min_year
    3. Handle duplicates (keep TOT for traded players)
    4. Calculate fantasy points
    5. Merge advanced stats
    6. Select only model-relevant columns
    7. Output CSV/parquet
    """
    
    # Load data
    player_pg = load_player_per_game(player_per_game_path)
    player_adv = load_player_advanced(player_advanced_path)
    
    # Filter for modern era
    logger.info(f"Filtering for seasons >= {min_year}")
    player_pg = player_pg[player_pg['season'] >= min_year].copy()
    player_adv = player_adv[player_adv['season'] >= min_year].copy()
    
    # Filter for minimum games played
    logger.info(f"Filtering for players with >= {min_games} games")
    player_pg = player_pg[player_pg['g'] >= min_games].copy()
    
    # Handle traded players: keep TOT row (total across teams)
    logger.info("Handling traded players (keeping TOT rows)")
    
    player_season_counts = player_pg.groupby(['player', 'season']).size()
    multi_team_players = player_season_counts[player_season_counts > 1].index
    
    mask_multi = player_pg.set_index(['player', 'season']).index.isin(multi_team_players)
    
    df_single = player_pg[~mask_multi].copy()
    df_multi = player_pg[mask_multi].copy()
    
    if 'team' in df_multi.columns:
        df_multi = df_multi[df_multi['team'] == 'TOT'].copy()
    
    player_pg = pd.concat([df_single, df_multi], ignore_index=True)
    
    # Calculate fantasy points per game
    logger.info("Calculating fantasy points per game")
    player_pg['fppg'] = calculate_fantasy_points(player_pg)
    
    # Merge with advanced stats
    logger.info("Merging with advanced stats")
    merge_cols = ['player', 'season']
    
    adv_cols = ['player', 'season', 'per', 'usg_percent', 'ws', 'bpm', 'vorp']
    adv_cols = [c for c in adv_cols if c in player_adv.columns]
    
    df = player_pg.merge(
        player_adv[adv_cols],
        on=merge_cols,
        how='left',
        suffixes=('', '_adv')
    )
    
    # Create normalized player name for joining with NBA API data
    df['player_normalized'] = df['player'].str.lower().str.strip()
    
    # Select only columns needed for model
    # Identifiers (for joining)
    id_cols = ['player_id', 'player_normalized', 'season', 'team', 'pos']
    
    # Features (for model)
    feature_cols = [
        # Per Game Stats
        'pts_per_game', 'ast_per_game', 'trb_per_game',
        'stl_per_game', 'blk_per_game', 'tov_per_game', 

        'fg_percent', 'ft_percent', 'x3p_percent', # shooting efficiency
        'mp_per_game', 'g',  # games played, minutes context 
        'fppg',  # calculated baseline fantasy points per game
        'per',   # player efficiency rating from advanced stats
         
        # 'ts_percent', 'usg_percent', # Removing these for now as they may introduce noise
    ]
    
    # Keep only columns that exist
    keep_cols = [c for c in id_cols + feature_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    logger.info(f"Final columns: {list(df.columns)}")
    
    # Output
    _save_dataframe(df, output_path)
    
    return df


def process_teams(
    team_stats_path: str | Path,
    team_summaries_path: str | Path,
    output_path: str | Path,
    min_year: int = 2022,
) -> pd.DataFrame:
    """
    Process team stats for opponent matchup features.
    
    Merges:
    - Team Stats Per Game: pts_per_game
    - Team Summaries: pace, o_rtg, d_rtg
    
    Output columns:
    - abbreviation (for joining with game log matchups)
    - season
    - team_pace, team_ortg, team_drtg
    - team_pts_per_game
    """
    
    logger.info(f"Processing team stats...")
    
    # Load Team Stats Per Game
    logger.info(f"Loading Team Stats Per Game from {team_stats_path}")
    df_stats = pd.read_csv(team_stats_path)
    df_stats.columns = df_stats.columns.str.lower().str.strip()
    
    # Load Team Summaries
    logger.info(f"Loading Team Summaries from {team_summaries_path}")
    df_summaries = pd.read_csv(team_summaries_path)
    df_summaries.columns = df_summaries.columns.str.lower().str.strip()
    
    # Filter for modern era
    df_stats['season'] = pd.to_numeric(df_stats['season'], errors='coerce')
    df_summaries['season'] = pd.to_numeric(df_summaries['season'], errors='coerce')
    
    df_stats = df_stats[df_stats['season'] >= min_year].copy()
    df_summaries = df_summaries[df_summaries['season'] >= min_year].copy()

    # Remove league average rows (empty team/abbreviation)
    if 'abbreviation' in df_stats.columns:
        df_stats = df_stats[df_stats['abbreviation'].notna() & (df_stats['abbreviation'] != '')].copy()
    if 'abbreviation' in df_summaries.columns:
        df_summaries = df_summaries[df_summaries['abbreviation'].notna() & (df_summaries['abbreviation'] != '')].copy()
    
    # Select columns from each source
    stats_cols = ['abbreviation', 'season', 'pts_per_game']
    stats_cols = [c for c in stats_cols if c in df_stats.columns]
    df_stats = df_stats[stats_cols].copy()
    
    summaries_cols = ['abbreviation', 'season', 'pace', 'o_rtg', 'd_rtg']
    summaries_cols = [c for c in summaries_cols if c in df_summaries.columns]
    df_summaries = df_summaries[summaries_cols].copy()
    
    # Merge on abbreviation + season
    df = df_stats.merge(
        df_summaries,
        on=['abbreviation', 'season'],
        how='outer'
    )
    
    # Convert to numeric
    for col in ['pts_per_game', 'pace', 'o_rtg', 'd_rtg']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename for clarity when joined as opponent stats
    rename_map = {
        'abbreviation': 'team',  # standardize to 'team' for joins
        'pace': 'team_pace',
        'o_rtg': 'team_ortg', 
        'd_rtg': 'team_drtg',
        'pts_per_game': 'team_pts_per_game',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Drop any duplicates
    df = df.drop_duplicates(subset=['team', 'season'])
    
    logger.info(f"Team stats columns: {list(df.columns)}")
    logger.info(f"Team-seasons processed: {len(df)}")
    
    # Output
    _save_dataframe(df, output_path)
    
    return df


def _save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save dataframe to CSV or parquet based on extension."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving to {output_path} (shape: {df.shape})")
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Kaggle NBA data")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory containing Kaggle CSVs")
    parser.add_argument("--output-dir", type=str, default="./data/processed",
                        help="Output directory for processed files")
    parser.add_argument("--min-year", type=int, default=2018,
                        help="Minimum season year to include")
    parser.add_argument("--min-games", type=int, default=10,
                        help="Minimum games played to include")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Process players
    print("\n" + "="*50)
    print("PROCESSING PLAYERS")
    print("="*50)
    
    df_players = process_players(
        player_per_game_path=data_dir / "Player Per Game.csv",
        player_advanced_path=data_dir / "Advanced.csv",
        output_path=output_dir / "processed_players.csv",
        min_year=args.min_year,
        min_games=args.min_games,
    )
    
    print(f"\nProcessed {len(df_players)} player-seasons")
    print(f"Columns: {list(df_players.columns)}")
    print(f"\nSample data:")
    print(df_players[['player_normalized', 'season', 'pts_per_game', 'fppg']].head(10))
    print(f"\nData types:\n{df_players.dtypes}")
    
    # Process teams
    print("\n" + "="*50)
    print("PROCESSING TEAMS")
    print("="*50)
    
    df_teams = process_teams(
        team_stats_path=data_dir / "Team Stats Per Game.csv",
        team_summaries_path=data_dir / "Team Summaries.csv",
        output_path=output_dir / "processed_teams.csv",
        min_year=args.min_year,
    )
    
    if not df_teams.empty:
        print(f"\nProcessed {len(df_teams)} team-seasons")
        print(f"Columns: {list(df_teams.columns)}")
        print(f"\nSample data:")
        print(df_teams.head(10))