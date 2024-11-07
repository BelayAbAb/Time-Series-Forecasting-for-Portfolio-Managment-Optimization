import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Step 1: Load Data and Preprocess
tickers = ['TSLA', 'BND', 'SPY']
start_date = '2015-01-01'
end_date = '2024-12-31'

# Fetch historical data for Tesla
data = yf.download(tickers, start=start_date, end=end_date)

# Check if the data contains a MultiIndex
print(data.head())  # to see how the columns are structured

# Access Tesla's adjusted close prices (adjusted close is under 'Adj Close')
tsla_data = data['Adj Close']['TSLA']

# Split data into training and test sets (80% training, 20% test)
train_size = int(len(tsla_data) * 0.8)
train, test = tsla_data[:train_size], tsla_data[train_size:]

# Ensure the report directory exists, create if not
report_dir = r'C:\Users\User\Desktop\10Acadamy\Week 11\Time Series Forecasting for Portfolio Managment Optimization\Report'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

# --- Plot and Save Stock Price ---
# Plot Tesla stock price and save as JPG
plt.figure(figsize=(12, 6))
tsla_data.plot(title="Tesla Stock Price (Adjusted Close)", label='TSLA', color='blue')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "tesla_stock_price.jpg"), format="jpg")
plt.close()

# --- ARIMA Model ---
# Build ARIMA model using auto_arima to find the optimal parameters (p, d, q)
model_arima = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
print(f"ARIMA Best Parameters: {model_arima.order}")

# Fit the ARIMA model
model_arima.fit(train)

# Forecast using the ARIMA model
forecast_arima = model_arima.predict(n_periods=len(test))

# Evaluate the ARIMA model performance
mae_arima = mean_absolute_error(test, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))
mape_arima = np.mean(np.abs((test - forecast_arima) / test)) * 100

print(f"ARIMA MAE: {mae_arima:.2f}")
print(f"ARIMA RMSE: {rmse_arima:.2f}")
print(f"ARIMA MAPE: {mape_arima:.2f}%")

# --- Plot ARIMA Forecast ---
# Plot ARIMA forecast and save as JPG
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast_arima, label='ARIMA Forecast', color='red')
plt.title('ARIMA Forecast vs Actual Tesla Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "arima_forecast.jpg"), format="jpg")
plt.close()

# --- SARIMA Model ---
# Build SARIMA model (seasonal ARIMA)
sarima_model = SARIMAX(train, order=(5, 1, 0), seasonal_order=(1, 1, 0, 5))
sarima_results = sarima_model.fit(disp=False)

# Forecast using the SARIMA model
forecast_sarima = sarima_results.forecast(steps=len(test))

# Evaluate the SARIMA model performance
mae_sarima = mean_absolute_error(test, forecast_sarima)
rmse_sarima = np.sqrt(mean_squared_error(test, forecast_sarima))
mape_sarima = np.mean(np.abs((test - forecast_sarima) / test)) * 100

print(f"SARIMA MAE: {mae_sarima:.2f}")
print(f"SARIMA RMSE: {rmse_sarima:.2f}")
print(f"SARIMA MAPE: {mape_sarima:.2f}%")

# --- Plot SARIMA Forecast ---
# Plot SARIMA forecast and save as JPG
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast_sarima, label='SARIMA Forecast', color='green')
plt.title('SARIMA Forecast vs Actual Tesla Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "sarima_forecast.jpg"), format="jpg")
plt.close()

# --- LSTM Model ---
# LSTM requires the data to be scaled between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Prepare data for LSTM model (using 60 timesteps as input for each prediction)
def create_lstm_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_lstm_dataset(train_scaled)
X_test, y_test = create_lstm_dataset(test_scaled)

# Reshape X to be [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the LSTM model
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict using the LSTM model
predicted_lstm = model_lstm.predict(X_test)

# Invert scaling
predicted_lstm = scaler.inverse_transform(predicted_lstm)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the LSTM model performance
mae_lstm = mean_absolute_error(y_test, predicted_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, predicted_lstm))
mape_lstm = np.mean(np.abs((y_test - predicted_lstm) / y_test)) * 100

print(f"LSTM MAE: {mae_lstm:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")
print(f"LSTM MAPE: {mape_lstm:.2f}%")

# --- Plot LSTM Forecast ---
# Plot LSTM forecast and save as JPG
plt.figure(figsize=(12, 6))
plt.plot(test.index[60:], y_test, label='Test')
plt.plot(test.index[60:], predicted_lstm, label='LSTM Forecast', color='purple')
plt.title('LSTM Forecast vs Actual Tesla Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, "lstm_forecast.jpg"), format="jpg")
plt.close()
