import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Directory where images will be saved
save_dir = r'C:\Users\User\Desktop\Time_Series_Forecasting_Outputs'
os.makedirs(save_dir, exist_ok=True)

# Fetch historical data for Tesla
ticker = 'TSLA'
start_date = '2015-01-01'
end_date = '2024-12-31'
data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess data: Use 'Adj Close' price and resample to monthly data
data = data[['Adj Close']].resample('M').last()  # Resample to monthly data

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Helper function to evaluate model performance
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# ---- ARIMA Model ----
def arima_model(train, test):
    # Fit ARIMA model (Use auto_arima to select best (p, d, q))
    model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
    print(f"Best ARIMA model parameters: {model.order}")
    
    # Forecast the stock prices
    forecast = model.predict(n_periods=len(test))
    
    # Evaluate the model
    mae, rmse, mape = evaluate_model(test, forecast)
    print(f"ARIMA Model Evaluation: MAE={mae}, RMSE={rmse}, MAPE={mape}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='ARIMA Forecast', linestyle='--')
    plt.title('ARIMA Model Forecasting (Monthly Data)')
    plt.legend()
    
    # Save the plot to a .jpg file
    plt.savefig(os.path.join(save_dir, 'ARIMA_Forecast_Monthly.jpg'), format='jpg')
    plt.close()

# ---- SARIMA Model ----
def sarima_model(train, test):
    # Fit SARIMA model (seasonal=True for capturing seasonality)
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Monthly seasonality (12 months)
    model_fit = model.fit(disp=False)
    
    # Forecast the stock prices
    forecast = model_fit.forecast(len(test))
    
    # Evaluate the model
    mae, rmse, mape = evaluate_model(test, forecast)
    print(f"SARIMA Model Evaluation: MAE={mae}, RMSE={rmse}, MAPE={mape}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, forecast, label='SARIMA Forecast', linestyle='--')
    plt.title('SARIMA Model Forecasting (Monthly Data)')
    plt.legend()
    
    # Save the plot to a .jpg file
    plt.savefig(os.path.join(save_dir, 'SARIMA_Forecast_Monthly.jpg'), format='jpg')
    plt.close()

# ---- LSTM Model ----
def lstm_model(train, test):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # Prepare the data for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    time_step = 12  # Use last 12 months to predict the next month (since data is monthly)
    X_train, y_train = create_dataset(train_scaled, time_step)
    X_test, y_test = create_dataset(test_scaled, time_step)
    
    # Reshape data for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Predict with LSTM
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Evaluate the model
    mae, rmse, mape = evaluate_model(test[time_step:].values, predicted_prices)
    print(f"LSTM Model Evaluation: MAE={mae}, RMSE={rmse}, MAPE={mape}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='True Prices')
    plt.plot(test.index[time_step:], predicted_prices, label='LSTM Forecast', linestyle='--')
    plt.title('LSTM Model Forecasting (Monthly Data)')
    plt.legend()
    
    # Save the plot to a .jpg file
    plt.savefig(os.path.join(save_dir, 'LSTM_Forecast_Monthly.jpg'), format='jpg')
    plt.close()

# ---- Run the models ----
print("Training and evaluating ARIMA model...")
arima_model(train, test)

print("Training and evaluating SARIMA model...")
sarima_model(train, test)

print("Training and evaluating LSTM model...")
lstm_model(train.values, test.values)

