import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

# Function to prepare data for LSTM/GRU
def prepare_lstm_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# Function to build LSTM model
def build_lstm_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build GRU model
def build_gru_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to plot results
def plot_results(train, test, predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    
    # Adjust predictions to align with test data indices
    predictions_index = test.index[:len(predictions)]
    plt.plot(predictions_index, predictions, label=f'{model_name} Predictions', color='red')
    
    plt.legend()
    plt.show()

# Function to compute accuracy metrics
def compute_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

# Main function
def main():
    # Load data
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60
    X, y, scaler = prepare_lstm_data(data, seq_length)
    X_train, y_train = X[:train_size], y[:train_size]  # Use full training set
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]  # Use full test set

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build and train LSTM model
    lstm_model = build_lstm_model(seq_length)
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Build and train GRU model
    gru_model = build_gru_model(seq_length)
    gru_model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Predictions from LSTM and GRU models
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)

    # Inverse transform predictions to original scale
    lstm_pred = scaler.inverse_transform(lstm_pred)
    gru_pred = scaler.inverse_transform(gru_pred)

    # Ensemble predictions using exponential formula
    alpha = 0.5  # Weight parameter for exponential formula
    ensemble_pred = alpha * lstm_pred + (1 - alpha) * gru_pred

    # Scale back the y_test for accuracy computation
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Compute accuracy metrics for ensemble predictions
    rmse, mape = compute_accuracy(y_test_scaled, ensemble_pred)
    print(f'Ensemble RMSE: {rmse}')
    print(f'Ensemble MAPE: {mape * 100}%')

    # Plot results for LSTM, GRU, and Ensemble
  
    plot_results(train, test, ensemble_pred, 'Ensemble')

if __name__ == "__main__":
    main()
