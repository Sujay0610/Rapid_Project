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
    data = load_data('data/BTC-USD.csv', 'data/sentiment_analysis_output_advanced.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60

    # With sentiment data
    print("\nEvaluating models with sentiment data:")
    X, y, scaler = prepare_lstm_data(data, seq_length, use_sentiment=True)
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]
    
    num_features = X_test.shape[2]
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

    # Load trained LSTM model
    lstm_model = tf.keras.models.load_model('lstm_model.h5')

    # Load trained GRU model
    gru_model = tf.keras.models.load_model('gru_model.h5')

    # Predictions from LSTM and GRU models
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)

    # Load the saved scaler parameters
    scaler = MinMaxScaler()
    scaler.scale_ = np.load('scaler.npy')
    scaler.min_ = np.load('scaler_min.npy')
    scaler.data_min_ = scaler.min_
    scaler.data_max_ = scaler.min_ + scaler.scale_

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
    plot_results(train, test, ensemble_pred, 'Ensemble with Sentiment')

if __name__ == "__main__":
    main()
