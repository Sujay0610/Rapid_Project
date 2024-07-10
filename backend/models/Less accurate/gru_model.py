import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

def prepare_gru_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_gru_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60
    X, y, scaler = prepare_gru_data(data, seq_length)
    X_train, y_train = X[:train_size-seq_length], y[:train_size-seq_length]
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    gru_model = build_gru_model(seq_length)
    gru_model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)

    gru_pred = gru_model.predict(X_test)
    gru_pred = scaler.inverse_transform(gru_pred).flatten()

    test_true = scaler.inverse_transform(y[train_size-seq_length:])

    gru_mse = mean_squared_error(test_true, gru_pred.reshape(-1, 1))
    print(f'GRU Model MSE: {gru_mse}')

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(test_true):], test_true, label='True Values')
    plt.plot(data.index[-len(gru_pred):], gru_pred, label='GRU Predictions')
    plt.title('BTC-USD Closing Price Prediction with GRU')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()
