import pandas as pd
import numpy as np
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

# Main function
def main():
    # Load data
    data = load_data('data/BTC-USD.csv', 'data/sentiment_analysis_output_advanced.csv')
    train_size = int(len(data) * 0.8)
    train = data.iloc[:train_size]

    seq_length = 60

    # With sentiment data
    print("\nTraining models with sentiment data:")
    X, y, scaler = prepare_lstm_data(data, seq_length, use_sentiment=True)
    X_train, y_train = X[:train_size], y[:train_size]
    
    num_features = X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))

    # Build and train LSTM model
    lstm_model = build_lstm_model(seq_length, num_features)
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=50)
    lstm_model.save('lstm_model.h5')

    # Build and train GRU model
    gru_model = build_gru_model(seq_length, num_features)
    gru_model.fit(X_train, y_train, batch_size=1, epochs=50)
    gru_model.save('gru_model.h5')

    # Save the scaler for later use
    np.save('scaler.npy', scaler.scale_)
    np.save('scaler_min.npy', scaler.min_)

if __name__ == "__main__":
    main()
