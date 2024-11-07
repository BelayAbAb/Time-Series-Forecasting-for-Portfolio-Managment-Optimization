import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
import os

# Define tickers and data fetching parameters
tickers = ['TSLA', 'BND', 'SPY']
start_date = '2015-01-01'
end_date = '2024-12-31'

# Directory where images will be saved
save_dir = r'C:\Users\User\Desktop\10Acadamy\Week 11\Time Series Forecasting for Portfolio Managment Optimization\Report'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Fetch historical data
data = yf.download(tickers, start=start_date, end=end_date)

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Clean data by filling or removing missing values (if any)
data = data.fillna(method='ffill')  # forward fill

# Check data types
print("\nData Types:\n", data.dtypes)

# Inspect the first few rows
print("\nFirst few rows:\n", data.head())

# ---- EDA and Visualization ----

# 1. Plotting the closing price of TSLA
plt.figure(figsize=(12, 6))
data['Adj Close']['TSLA'].plot(title="Tesla Stock Price (Adjusted Close)", label='TSLA', color='blue')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
# Save the plot as JPG in the specified directory
plt.savefig(os.path.join(save_dir, 'TSLA_Stock_Price.jpg'), format='jpg')
plt.close()  # Close the figure to free memory

# 2. Calculate daily returns and plot
tsla_returns = data['Adj Close']['TSLA'].pct_change().dropna()
plt.figure(figsize=(12, 6))
tsla_returns.plot(title="Tesla Daily Returns", label='TSLA Daily Returns', color='red')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
# Save the plot as JPG in the specified directory
plt.savefig(os.path.join(save_dir, 'TSLA_Daily_Returns.jpg'), format='jpg')
plt.close()

# 3. Calculate and plot rolling volatility (252-day rolling window, which is roughly one trading year)
rolling_volatility = tsla_returns.rolling(window=252).std() * np.sqrt(252)  # Annualized volatility
plt.figure(figsize=(12, 6))
rolling_volatility.plot(title="Tesla Annualized Rolling Volatility", label='Rolling Volatility (252 days)', color='green')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
# Save the plot as JPG in the specified directory
plt.savefig(os.path.join(save_dir, 'TSLA_Rolling_Volatility.jpg'), format='jpg')
plt.close()

# ---- Outlier Detection ----
# Detect extreme daily returns using z-scores (values above or below 2 standard deviations)
z_scores = (tsla_returns - tsla_returns.mean()) / tsla_returns.std()
outliers = z_scores[abs(z_scores) > 2]  # Outliers defined as 2 standard deviations away
print("\nOutliers in daily returns:\n", outliers)

# ---- Seasonality and Trend Decomposition ----
# Decompose the time series into trend, seasonal, and residual components using seasonal decomposition
tsla_prices = data['Adj Close']['TSLA']
decomposition = seasonal_decompose(tsla_prices, model='multiplicative', period=252)  # 252 trading days per year

# Plot the decomposition
plt.figure(figsize=(12, 8))
decomposition.plot()
# Save the plot as JPG in the specified directory
plt.savefig(os.path.join(save_dir, 'TSLA_Time_Series_Decomposition.jpg'), format='jpg')
plt.close()

# ---- Volatility and Risk Analysis ----

# 1. Value at Risk (VaR) at 95% confidence level
# Calculate the 5th percentile of the return distribution for the 95% confidence level
VaR_95 = tsla_returns.quantile(0.05)
print(f"\nValue at Risk (VaR) at 95% confidence: {VaR_95 * 100:.2f}%")

# 2. Sharpe Ratio (assuming risk-free rate is 0)
# Sharpe Ratio = (Mean Return - Risk-Free Rate) / Standard Deviation of Return
mean_return = tsla_returns.mean() * 252  # Annualize the mean return
std_dev_return = tsla_returns.std() * np.sqrt(252)  # Annualize the standard deviation
sharpe_ratio = mean_return / std_dev_return
print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")

# ---- Distribution of Returns ----

# Plot the distribution of Tesla's returns
plt.figure(figsize=(10, 6))
sns.histplot(tsla_returns, bins=100, kde=True, color='purple')
plt.title("Distribution of Tesla Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
# Save the plot as JPG in the specified directory
plt.savefig(os.path.join(save_dir, 'TSLA_Return_Distribution.jpg'), format='jpg')
plt.close()

# ---- Conclusion ----
# The plots and statistics provide insights into Tesla's stock price behavior, volatility, risk, and returns.
