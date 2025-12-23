"""
Data Loader - Download stock data from Yahoo Finance
Fixed version: Handles yfinance MultiIndex columns correctly
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from config import Config

def fetch_stock_data(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date (default: 2010-01-01)
        end_date: End date (default: today)
        use_cache: Use cached data if available
    
    Returns:
        DataFrame with OHLCV data
    """
    
    if start_date is None:
        start_date = Config.START_DATE
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Cache file path
    cache_dir = Config.DATA_DIR / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_data.csv"
    
    # Check cache
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) > 100:
                print(f"   📦 Using cached data: {len(df)} rows")
                return df
        except Exception:
            pass
    
    # Download fresh data
    try:
        print(f"   📥 Downloading {symbol}...")
        
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"   ❌ No data returned for {symbol}")
            return pd.DataFrame()
        
        # Fix column names - handle both single and MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            # If MultiIndex, flatten it
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            # Regular columns
            df.columns = [col.lower() for col in df.columns]
        
        # Keep only required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if not available_cols:
            print(f"   ❌ {symbol}: No valid columns found")
            return pd.DataFrame()
        
        df = df[available_cols]
        
        # Remove NaN rows
        df = df.dropna()
        
        if len(df) < 100:
            print(f"   ❌ {symbol}: Only {len(df)} rows (need at least 100)")
            return pd.DataFrame()
        
        # Save to cache
        try:
            df.to_csv(cache_file)
        except Exception:
            pass
        
        print(f"   ✅ Downloaded: {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")
        return pd.DataFrame()


def get_stock_list() -> list:
    """Get list of supported stocks"""
    return Config.SUPPORTED_STOCKS


if __name__ == "__main__":
    # Test download
    print("Testing data loader...")
    df = fetch_stock_data('AAPL')
    print(f"Downloaded {len(df)} rows")
    if not df.empty:
        print(df.head())