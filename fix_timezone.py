#!/usr/bin/env python3
"""
Quick script to test timezone fix
"""

import pandas as pd
from src.data_loader import fetch_stock_data

print("Testing timezone fix...")

# Test with AAPL
df = fetch_stock_data('AAPL', use_cache=True)
print(f"\n✅ Loaded AAPL: {len(df)} rows")
print(f"   Index type: {type(df.index)}")
print(f"   First index value type: {type(df.index[0])}")

# Force convert to DatetimeIndex and handle timezone
if not isinstance(df.index, pd.DatetimeIndex):
    print(f"   Converting to DatetimeIndex...")
    df.index = pd.to_datetime(df.index, utc=True)  # Parse as UTC first
    print(f"   ✅ Converted to DatetimeIndex")

# Check for timezone
if df.index.tz is not None:
    print(f"   Has timezone: {df.index.tz}")
    print(f"   Removing timezone...")
    df.index = df.index.tz_localize(None)
    print(f"   ✅ Timezone removed")
else:
    print(f"   No timezone present")

print(f"\n   Final index type: {type(df.index)}")
print(f"   First date: {df.index[0]}")
print(f"   Last date: {df.index[-1]}")

# Test date filtering
train_end = pd.to_datetime("2018-12-31")
val_end = pd.to_datetime("2021-12-31")
test_end = pd.to_datetime("2024-12-31")

print(f"\n   Comparing {type(df.index[0])} with {type(train_end)}")

try:
    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[(df.index > val_end) & (df.index <= test_end)]

    print(f"\n📊 Split results:")
    print(f"   Train: {len(train_df)} rows (up to 2018-12-31)")
    print(f"   Val:   {len(val_df)} rows (2019-2021)")
    print(f"   Test:  {len(test_df)} rows (2022-2024)")

    if len(train_df) > 0 and len(val_df) > 0:
        print(f"\n✅ Timezone fix successful!")
        print(f"   Train dates: {train_df.index[0]} to {train_df.index[-1]}")
        print(f"   Val dates:   {val_df.index[0]} to {val_df.index[-1]}")
        if len(test_df) > 0:
            print(f"   Test dates:  {test_df.index[0]} to {test_df.index[-1]}")
    else:
        print(f"\n⚠️  Warning: Some splits are empty")
        print(f"   Check if data covers the expected date ranges")
        
except Exception as e:
    print(f"\n❌ Comparison failed: {e}")
    print(f"   This means we need a different approach")