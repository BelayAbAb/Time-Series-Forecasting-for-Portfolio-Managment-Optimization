import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
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

# Clean data by filling any missing values
data = data.fillna(method='ffill')  # forward fill

# Function to plot and save a grid comparison
def plot_and_save_comparison(data, comparison_type, save_name):
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot TSLA
    axs[0, 0].plot(data['Adj Close']['TSLA'], color='blue', label="TSLA")
    axs[0, 0].set_title('TSLA ' + comparison_type)
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel(comparison_type)
    axs[0, 0].legend()

    # Plot BND
    axs[0, 1].plot(data['Adj Close']['BND'], color='green', label="BND")
    axs[0, 1].set_title('BND ' + comparison_type)
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel(comparison_type)
    axs[0, 1].legend()

    # Plot SPY
    axs[1, 0].plot(data['Adj Close']['SPY'], color='orange', label="SPY")
    axs[1, 0].set_title('SPY ' + comparison_type)
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel(comparison_type)
    axs[1, 0].legend()

    # Plot Comparison (TSLA, BND, SPY)
    axs[1, 1].plot(data['Adj Close']['TSLA'], color='blue', label="TSLA", linewidth=1.5)
    axs[1, 1].plot(data['Adj Close']['BND'], color='green', label="BND", linewidth=1.5)
    axs[1, 1].plot(data['Adj Close']['SPY'], color='orange', label="SPY", linewidth=1.5)
    axs[1, 1].set_title(f'Comparison of {comparison_type} for TSLA, BND, SPY')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel(comparison_type)
    axs[1, 1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name), format='jpg')
    plt.close()

# ---- Stock Price Comparison (Adjusted Close) ----
plot_and_save_comparison(data, "Stock Price", "Stock_Price_Comparison_TSLA_BND_SPY.jpg")

# ---- Daily Returns Comparison ----
# Calculate daily returns
daily_returns = data['Adj Close'].pct_change().dropna()

def plot_daily_returns_comparison():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Plot TSLA Daily Returns
    axs[0, 0].plot(daily_returns['TSLA'], color='blue', label="TSLA Daily Returns")
    axs[0, 0].set_title('TSLA Daily Returns')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Daily Return')
    axs[0, 0].legend()

    # Plot BND Daily Returns
    axs[0, 1].plot(daily_returns['BND'], color='green', label="BND Daily Returns")
    axs[0, 1].set_title('BND Daily Returns')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Daily Return')
    axs[0, 1].legend()

    # Plot SPY Daily Returns
    axs[1, 0].plot(daily_returns['SPY'], color='orange', label="SPY Daily Returns")
    axs[1, 0].set_title('SPY Daily Returns')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Daily Return')
    axs[1, 0].legend()

    # Plot Comparison of Daily Returns (TSLA, BND, SPY)
    axs[1, 1].plot(daily_returns['TSLA'], color='blue', label="TSLA", linewidth=1.5)
    axs[1, 1].plot(daily_returns['BND'], color='green', label="BND", linewidth=1.5)
    axs[1, 1].plot(daily_returns['SPY'], color='orange', label="SPY", linewidth=1.5)
    axs[1, 1].set_title('Comparison of Daily Returns (TSLA, BND, SPY)')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Daily Return')
    axs[1, 1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Daily_Returns_Comparison_TSLA_BND_SPY.jpg'), format='jpg')
    plt.close()

plot_daily_returns_comparison()

# ---- Return Distribution Comparison ----
def plot_return_distribution_comparison():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Plot TSLA Return Distribution
    sns.histplot(daily_returns['TSLA'], bins=100, kde=True, color='blue', ax=axs[0, 0])
    axs[0, 0].set_title('TSLA Return Distribution')
    axs[0, 0].set_xlabel('Daily Return')
    axs[0, 0].set_ylabel('Frequency')

    # Plot BND Return Distribution
    sns.histplot(daily_returns['BND'], bins=100, kde=True, color='green', ax=axs[0, 1])
    axs[0, 1].set_title('BND Return Distribution')
    axs[0, 1].set_xlabel('Daily Return')
    axs[0, 1].set_ylabel('Frequency')

    # Plot SPY Return Distribution
    sns.histplot(daily_returns['SPY'], bins=100, kde=True, color='orange', ax=axs[1, 0])
    axs[1, 0].set_title('SPY Return Distribution')
    axs[1, 0].set_xlabel('Daily Return')
    axs[1, 0].set_ylabel('Frequency')

    # Plot Comparison of Return Distributions (TSLA, BND, SPY)
    sns.histplot(daily_returns['TSLA'], bins=100, kde=True, color='blue', ax=axs[1, 1], label="TSLA")
    sns.histplot(daily_returns['BND'], bins=100, kde=True, color='green', ax=axs[1, 1], label="BND")
    sns.histplot(daily_returns['SPY'], bins=100, kde=True, color='orange', ax=axs[1, 1], label="SPY")
    axs[1, 1].set_title('Comparison of Return Distributions (TSLA, BND, SPY)')
    axs[1, 1].set_xlabel('Daily Return')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Return_Distribution_Comparison_TSLA_BND_SPY.jpg'), format='jpg')
    plt.close()

plot_return_distribution_comparison()

# ---- Rolling Volatility Comparison ----
def plot_rolling_volatility_comparison():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Calculate rolling volatility for each asset
    rolling_volatility_tsla = daily_returns['TSLA'].rolling(window=252).std() * np.sqrt(252)
    rolling_volatility_bnd = daily_returns['BND'].rolling(window=252).std() * np.sqrt(252)
    rolling_volatility_spy = daily_returns['SPY'].rolling(window=252).std() * np.sqrt(252)

    # Plot TSLA Rolling Volatility
    axs[0, 0].plot(rolling_volatility_tsla, color='blue', label="TSLA Rolling Volatility")
    axs[0, 0].set_title('TSLA Rolling Volatility')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Volatility')
    axs[0, 0].legend()

    # Plot BND Rolling Volatility
    axs[0, 1].plot(rolling_volatility_bnd, color='green', label="BND Rolling Volatility")
    axs[0, 1].set_title('BND Rolling Volatility')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Volatility')
    axs[0, 1].legend()

    # Plot SPY Rolling Volatility
    axs[1, 0].plot(rolling_volatility_spy, color='orange', label="SPY Rolling Volatility")
    axs[1, 0].set_title('SPY Rolling Volatility')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Volatility')
    axs[1, 0].legend()

    # Plot Comparison of Rolling Volatility (TSLA, BND, SPY)
    axs[1, 1].plot(rolling_volatility_tsla, color='blue', label="TSLA", linewidth=1.5)
    axs[1, 1].plot(rolling_volatility_bnd, color='green', label="BND", linewidth=1.5)
    axs[1, 1].plot(rolling_volatility_spy, color='orange', label="SPY", linewidth=1.5)
    axs[1, 1].set_title('Comparison of Rolling Volatility (TSLA, BND, SPY)')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Volatility')
    axs[1, 1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Rolling_Volatility_Comparison_TSLA_BND_SPY.jpg'), format='jpg')
    plt.close()

plot_rolling_volatility_comparison()

# ---- Time Series Decomposition Additive Comparison ----
def plot_time_series_decomposition_comparison():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Decompose time series for each asset
    tsla_decomposition = seasonal_decompose(data['Adj Close']['TSLA'], model='additive', period=252)
    bnd_decomposition = seasonal_decompose(data['Adj Close']['BND'], model='additive', period=252)
    spy_decomposition = seasonal_decompose(data['Adj Close']['SPY'], model='additive', period=252)

    # Plot TSLA Time Series Decomposition
    tsla_decomposition.plot(ax=axs[0, 0])
    axs[0, 0].set_title('TSLA Time Series Decomposition (Additive)')

    # Plot BND Time Series Decomposition
    bnd_decomposition.plot(ax=axs[0, 1])
    axs[0, 1].set_title('BND Time Series Decomposition (Additive)')

    # Plot SPY Time Series Decomposition
    spy_decomposition.plot(ax=axs[1, 0])
    axs[1, 0].set_title('SPY Time Series Decomposition (Additive)')

    # Plot Comparison of Time Series Decompositions
    axs[1, 1].plot(tsla_decomposition.trend, color='blue', label="TSLA Trend")
    axs[1, 1].plot(bnd_decomposition.trend, color='green', label="BND Trend")
    axs[1, 1].plot(spy_decomposition.trend, color='orange', label="SPY Trend")
    axs[1, 1].set_title('Comparison of Time Series Decompositions (TSLA, BND, SPY)')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Trend')
    axs[1, 1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Time_Series_Decomposition_Comparison_TSLA_BND_SPY.jpg'), format='jpg')
    plt.close()

plot_time_series_decomposition_comparison()

print("All plots have been saved successfully.")
