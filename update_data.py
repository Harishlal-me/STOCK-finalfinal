import yfinance as yf
import pandas as pd
from datetime import datetime

# List of stocks to update
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

# Folder to save CSVs
data_folder = "./data/"  # adjust if your CSVs are elsewhere

# Today's date
today = datetime.today().strftime("%Y-%m-%d")

for ticker in stocks:
    # Path to local CSV
    csv_file = f"{data_folder}{ticker}.csv"

    # Load existing CSV if exists
    try:
        df_existing = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        start_date = df_existing.index[-1]  # last available date
        print(f"{ticker}: Existing data until {start_date.date()}")
        # Start fetching from the next day
        start_date = (start_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        start_date = "2010-01-04"  # your default start
        print(f"{ticker}: No existing CSV found, starting from {start_date}")

    # Download new data from Yahoo Finance
    df_new = yf.download(ticker, start=start_date, end=today)

    if df_new.empty:
        print(f"{ticker}: No new data available.")
        continue

    # Combine with existing data if any
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
    else:
        df_combined = df_new

    # Save updated CSV
    df_combined.to_csv(csv_file)
    print(f"{ticker}: CSV updated! Last row: {df_combined.index[-1].date()}\n")
