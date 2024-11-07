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
data = data.ffill()  # forward fill

# Function to plot and save the comparison for time series decomposition (additive)
def plot_time_series_decomposition_comparison():
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Decompose the time series for each asset using additive model
    tsla_decomposition = seasonal_decompose(data['Adj Close']['TSLA'], model='additive', period=252)  # 252 trading days per year
    bnd_decomposition = seasonal_decompose(data['Adj Close']['BND'], model='additive', period=252)
    spy_decomposition = seasonal_decompose(data['Adj Close']['SPY'], model='additive', period=252)

    # Plot TSLA Time Series Decomposition
    axs[0, 0].plot(tsla_decomposition.observed, label='Observed', color='blue', linewidth=1.5)
    axs[0, 0].plot(tsla_decomposition.trend, label='Trend', color='orange', linewidth=1.5)
    axs[0, 0].plot(tsla_decomposition.seasonal, label='Seasonal', color='green', linewidth=1.5)
    axs[0, 0].plot(tsla_decomposition.resid, label='Residual', color='red', linewidth=1.5)
    axs[0, 0].set_title('TSLA Time Series Decomposition (Additive)')
    axs[0, 0].legend()

    # Plot BND Time Series Decomposition
    axs[0, 1].plot(bnd_decomposition.observed, label='Observed', color='blue', linewidth=1.5)
    axs[0, 1].plot(bnd_decomposition.trend, label='Trend', color='orange', linewidth=1.5)
    axs[0, 1].plot(bnd_decomposition.seasonal, label='Seasonal', color='green', linewidth=1.5)
    axs[0, 1].plot(bnd_decomposition.resid, label='Residual', color='red', linewidth=1.5)
    axs[0, 1].set_title('BND Time Series Decomposition (Additive)')
    axs[0, 1].legend()

    # Plot SPY Time Series Decomposition
    axs[1, 0].plot(spy_decomposition.observed, label='Observed', color='blue', linewidth=1.5)
    axs[1, 0].plot(spy_decomposition.trend, label='Trend', color='orange', linewidth=1.5)
    axs[1, 0].plot(spy_decomposition.seasonal, label='Seasonal', color='green', linewidth=1.5)
    axs[1, 0].plot(spy_decomposition.resid, label='Residual', color='red', linewidth=1.5)
    axs[1, 0].set_title('SPY Time Series Decomposition (Additive)')
    axs[1, 0].legend()

    # Plot Comparison of Time Series Decompositions (Trend component of each)
    axs[1, 1].plot(tsla_decomposition.trend, color='blue', label="TSLA Trend", linewidth=1.5)
    axs[1, 1].plot(bnd_decomposition.trend, color='green', label="BND Trend", linewidth=1.5)
    axs[1, 1].plot(spy_decomposition.trend, color='orange', label="SPY Trend", linewidth=1.5)
    axs[1, 1].set_title('Comparison of Time Series Decompositions (Trend) - TSLA, BND, SPY')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Trend')
    axs[1, 1].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Time_Series_Decomposition_Comparison_TSLA_BND_SPY.jpg'), format='jpg')
    plt.close()

# Run the Time Series Decomposition Comparison
plot_time_series_decomposition_comparison()

print("Time Series Decomposition comparison plot has been saved successfully.")
