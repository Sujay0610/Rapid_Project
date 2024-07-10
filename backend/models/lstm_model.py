#LSTM model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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

def build_lstm_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_results(train, test, predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    
    # Adjust predictions to align with test data indices
    predictions_index = test.index[:len(predictions)]
    plt.plot(predictions_index, predictions, label=f'{model_name} Predictions', color='red')
    
    plt.legend()
    plt.show()

def compute_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

def main():
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60
    X, y, scaler = prepare_lstm_data(data, seq_length)
    X_train, y_train = X[:train_size], y[:train_size]  # Use full training set
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]  # Use full test set

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = build_lstm_model(seq_length)
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

    lstm_pred = lstm_model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)

    # Scale back the y_test for accuracy computation
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mape = compute_accuracy(y_test_scaled, lstm_pred)
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    plot_results(train, test, lstm_pred, 'LSTM')

if __name__ == "__main__":
    main()

