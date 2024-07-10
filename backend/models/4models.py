import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

def prepare_lstm_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# Build LSTM model
def build_lstm_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train LSTM model
def train_lstm_model(X_train, y_train, seq_length):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    lstm_model = build_lstm_model(seq_length)
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)
    return lstm_model

# Build and train XGBoost model
def train_xgb_model(X_train, y_train):
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
    return xgb_model

# Build and train RandomForest model
def train_rf_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    return rf_model

# Ensemble predictions
def ensemble_predictions(models, X_test):
    predictions = []
    for model in models:
        if isinstance(model, tf.keras.models.Sequential):
            pred = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1))).flatten()
        else:
            pred = model.predict(X_test)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

# Calculate metrics
def compute_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

# Main function
def main():
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60
    X, y, scaler = prepare_lstm_data(data, seq_length)
    X_train, y_train = X[:train_size], y[:train_size]  # Use full training set
    X_test, y_test = X[train_size:], y[train_size:]    # Use full test set

    # Train individual models
    lstm_model = train_lstm_model(X_train, y_train, seq_length)
    xgb_model = train_xgb_model(X_train, y_train)
    rf_model = train_rf_model(X_train, y_train)

    # Make predictions
    lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))).flatten()
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Ensemble predictions
    models = [lstm_model, xgb_model, rf_model]
    ensemble_pred = ensemble_predictions(models, X_test.reshape(X_test.shape[0], -1))

    # Calculate metrics for ensemble predictions
    rmse, mape = compute_accuracy(y_test, ensemble_pred)
    print(f'Ensemble Model - RMSE: {rmse}')
    print(f'Ensemble Model - MAPE: {mape}%')

  # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')

# Prepare x-axis for ensemble predictions
    test_dates = test.index[:len(ensemble_pred)]

    plt.plot(test_dates, ensemble_pred, label='Ensemble Predictions', color='red')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
