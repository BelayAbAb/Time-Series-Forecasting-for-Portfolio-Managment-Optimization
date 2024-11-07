import yfinance as yf
import pandas as pd

# Define tickers for the assets
tickers = ['TSLA', 'BND', 'SPY']

# Set the period and start/end dates
start_date = '2015-01-01'
end_date = '2024-12-31'

# Fetch the historical data
data = yf.download(tickers, start=start_date, end=end_date)

# Print the first few rows of data to check structure
print(data.head())

# Extract Tesla's adjusted close data
tsla_data = data['Adj Close']['TSLA']
print(tsla_data.head())

# Calculate volatility for Tesla (using annualized rolling volatility)
tsla_returns = tsla_data.pct_change().dropna()
tsla_volatility = tsla_returns.rolling(window=252).std() * (252 ** 0.5)  # Annualized volatility

print("Tesla Volatility:")
print(tsla_volatility.tail())

# Save data to CSV for further analysis
data.to_csv('historical_financial_data.csv')
