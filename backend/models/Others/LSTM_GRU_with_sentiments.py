import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Function to load data
def load_data(price_file_path, sentiment_file_path):
    price_data = pd.read_csv(price_file_path)
    sentiment_data = pd.read_csv(sentiment_file_path)
    
    price_data['Date'] = pd.to_datetime(price_data['Date'], format='%m/%d/%y')
    price_data.set_index('Date', inplace=True)
    price_data = price_data.fillna(method='ffill')
    
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['created_at'], format='%Y-%m-%d')
    sentiment_data.set_index('Date', inplace=True)
    
    # Combine price data and sentiment scores using outer join
    data = price_data.join(sentiment_data[['sentiment_compound']], how='outer')
    
    # Fill missing sentiment scores with default value (e.g., 0)
    data['sentiment_compound'].fillna(0, inplace=True)
    
    # Fill any remaining NaN values in price data
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    
    return data

# Function to prepare data for LSTM/GRU
def prepare_lstm_data(data, seq_length, use_sentiment=True):
    if not use_sentiment:
        data = data.drop(columns=['sentiment_compound'])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])  # Assuming 'Close' is the first column
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# Function to build LSTM model
def build_lstm_model(seq_length, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build GRU model
def build_gru_model(seq_length, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, num_features)),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to plot results
def plot_results(train, test, predictions, future_dates, future_predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    plt.plot(test.index, predictions[:len(test)], label=f'{model_name} Predictions', color='red')
    plt.plot(future_dates, future_predictions, label=f'{model_name} Extended Predictions', color='green', linestyle='--')
    
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
    data = load_data('data/BTC-USD.csv', 'data/sentiment_analysis_output_new.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60

    # With sentiment data
    print("\nTraining and evaluating models with sentiment data:")
    X, y, scaler = prepare_lstm_data(data, seq_length, use_sentiment=True)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]
    
    num_features = X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

    # Build and train LSTM model
    lstm_model = build_lstm_model(seq_length, num_features)
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Build and train GRU model
    gru_model = build_gru_model(seq_length, num_features)
    gru_model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Predictions from LSTM and GRU models
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)

    # Inverse transform predictions to original scale
    lstm_pred = scaler.inverse_transform(np.hstack((lstm_pred, np.zeros((lstm_pred.shape[0], num_features - 1)))))[:, 0]
    gru_pred = scaler.inverse_transform(np.hstack((gru_pred, np.zeros((gru_pred.shape[0], num_features - 1)))))[:, 0]

    # Ensemble predictions using exponential formula
    alpha = 0.5
    ensemble_pred = alpha * lstm_pred + (1 - alpha) * gru_pred

    y_test_scaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1)))))[:, 0]

    rmse, mape = compute_accuracy(y_test_scaled, ensemble_pred)
    print(f'With Sentiment - Ensemble RMSE: {rmse}')
    print(f'With Sentiment - Ensemble MAPE: {mape * 100}%')

    # Plot results for LSTM, GRU, and Ensemble with sentiment data
    plot_results(train, test, ensemble_pred, [], [], 'Ensemble with Sentiment')

    # Extend predictions beyond test dataset
    horizon_days = 30  # Number of days to predict into the future
    future_dates = pd.date_range(test.index[-1], periods=horizon_days+1, freq='D')[1:]  # Start from the next day

    # Initialize future_X with the last sequence from X_test
    future_X = np.copy(X_test[-1])
    future_predictions = []

    for _ in range(horizon_days):
        # Predict using the last sequence in future_X
        lstm_future = lstm_model.predict(future_X[np.newaxis, :, :])
        gru_future = gru_model.predict(future_X[np.newaxis, :, :])

        # Inverse transform predictions
        lstm_future_inv = scaler.inverse_transform(np.hstack((lstm_future, np.zeros((1, num_features - 1)))))[:, 0]
        gru_future_inv = scaler.inverse_transform(np.hstack((gru_future, np.zeros((1, num_features - 1)))))[:, 0]

        # Ensemble prediction
        ensemble_future = alpha * lstm_future_inv + (1 - alpha) * gru_future_inv
        future_predictions.append(ensemble_future[0])

        # Update future_X for the next prediction
        new_feature_vector = np.hstack((lstm_future, np.zeros((1, num_features - 1))))  # Shape: (1, num_features)
        future_X = np.vstack((future_X[1:], new_feature_vector))  # Update with lstm_future (or gru_future)

    # Plot extended predictions
    plot_results(train, test, ensemble_pred, future_dates, future_predictions, 'Ensemble with Sentiment')

if __name__ == "__main__":
    main()
