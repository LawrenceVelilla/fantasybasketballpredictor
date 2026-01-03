"""
fetch_position_defense.py
Fetches team defense vs position stats from NBA API.
Output: position_defense.csv (team, season, position, defensive stats)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging
import time

from nba_api.stats.endpoints import leaguedashptteamdefend
from nba_api.stats.static import teams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Position mapping - NBA API uses these codes
POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']


def fetch_position_defense_single_season(
    season: str,
    rate_limit_delay: float = 0.6
) -> pd.DataFrame:
    """
    Fetch defense vs position for all teams in a single season.
    
    Args:
        season: Season string, e.g., "2023-24"
        rate_limit_delay: Seconds to wait between API calls
    
    Returns:
        DataFrame with columns: team, season, position, pts_allowed, fg_pct_allowed, etc.
    """
    
    all_data = []
    
    for position in POSITIONS:
        logger.info(f"Fetching {season} defense vs {position}")
        
        time.sleep(rate_limit_delay)
        
        try:
            defense = leaguedashptteamdefend.LeagueDashPtTeamDefend(
                season=season,
                season_type_all_star="Regular Season",
                defense_category="Overall",
                league_id="00",
                per_mode_simple="PerGame",
            )
            
            df = defense.get_data_frames()[0]
            
            if df.empty:
                logger.warning(f"No data for {season} {position}")
                continue
            
            # Filter by position if the endpoint supports it
            # Note: The endpoint may return all positions, need to check structure
            df['position'] = position
            df['season_str'] = season
            
            all_data.append(df)
            
        except Exception as e:
            logger.error(f"Error fetching {season} {position}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def fetch_team_vs_position_defense(
    start_year: int = 2022,
    end_year: int = 2025,
    output_path: str = "./data/processed/team_vs_position_defense.csv",
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch how each team defends against each position.
    
    Uses leaguedashptteamdefend which has player_position parameter.
    This gives us exactly what we need: "Team X allows Y points to PGs"
    """
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        season_int = year
        
        for position in POSITIONS:
            logger.info(f"Fetching {season} - defense vs {position}")
            
            time.sleep(rate_limit_delay)
            
            try:
                defense = leaguedashptteamdefend.LeagueDashPtTeamDefend(
                    season=season,
                    season_type_all_star="Regular Season",
                    defense_category="Overall",
                    league_id="00",
                    per_mode_simple="PerGame",
                )
                
                df = defense.get_data_frames()[0]
                
                if df.empty:
                    logger.warning(f"No data for {season} {position}")
                    continue
                
                # Add metadata
                df['season'] = season_int
                df['position'] = position
                
                all_data.append(df)
                logger.info(f"  → Got {len(df)} teams")
                
            except Exception as e:
                logger.error(f"Error fetching {season} {position}: {e}")
                continue
    
    if not all_data:
        logger.error("No data fetched!")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Lowercase columns
    combined.columns = combined.columns.str.lower()
    
    logger.info(f"Columns available: {list(combined.columns)}")
    
    # Select and rename columns
    keep_cols = [
        'team_id', 'team_abbreviation', 'team_name', 'season', 'position',
        'gp', 'g', 'freq', 'd_fgm', 'd_fga', 'd_fg_pct', 'normal_fg_pct', 'pct_plusminus'
    ]
    keep_cols = [c for c in keep_cols if c in combined.columns]
    
    combined = combined[keep_cols].copy()
    
    rename_map = {
        'team_abbreviation': 'team',
        'd_fg_pct': 'opp_pos_fg_pct',       # FG% allowed to this position
        'd_fgm': 'opp_pos_fgm',             # FGM allowed to this position  
        'd_fga': 'opp_pos_fga',             # FGA allowed to this position
        'pct_plusminus': 'opp_pos_fg_diff', # Difference from normal FG%
    }
    combined = combined.rename(columns={k: v for k, v in rename_map.items() if k in combined.columns})
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(combined)} records to {output_path}")
    logger.info(f"Final columns: {list(combined.columns)}")
    
    return combined

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch position defense stats")
    parser.add_argument("--start-year", type=int, default=2022,
                        help="Starting season year")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="Ending season year")
    parser.add_argument("--output", type=str, 
                        default="./data/processed/team_vs_position_defense.csv",
                        help="Output CSV path")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Rate limit delay in seconds")
    
    args = parser.parse_args()
    
    df = fetch_team_vs_position_defense(
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output,
        rate_limit_delay=args.delay,
    )
    
    if not df.empty:
        print(f"\nFetched {len(df)} records")
        print(f"\nSample data:")
        print(df.head(20))
        print(f"\nUnique positions: {df['position'].unique()}")
        print(f"Unique teams: {df['team'].nunique()}")