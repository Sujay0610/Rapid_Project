import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

def prepare_ann_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_ann_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_results(train, test, predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    plt.plot(test.index, predictions, label=f'{model_name} Predictions', color='red')
    plt.legend()
    plt.show()

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    seq_length = 60

    # Prepare data for the entire dataset
    X, y, scaler = prepare_ann_data(data, seq_length)

    # Split data into training and testing sets
    X_train, y_train = X[:train_size-seq_length], y[:train_size-seq_length]
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]

    ann_model = build_ann_model(seq_length)
    ann_model.fit(X_train, y_train, batch_size=1, epochs=5)

    ann_pred_scaled = ann_model.predict(X_test)
    ann_pred = scaler.inverse_transform(ann_pred_scaled).flatten()

    # Correctly align the test true values
    test_true = data['Close'].values[train_size:]

    # Ensure the lengths of predictions and true values match
    if len(ann_pred) != len(test_true):
        print("Warning: Lengths of ann_pred and test_true do not match!")
        min_len = min(len(ann_pred), len(test_true))
        ann_pred = ann_pred[:min_len]
        test_true = test_true[:min_len]

    ann_mse = mean_squared_error(test_true, ann_pred)
    print(f'ANN Model MSE: {ann_mse}')

    ann_mape = calculate_mape(test_true, ann_pred)
    print(f'ANN Model MAPE: {ann_mape}%')

    # Plotting results
    plot_results(data[:train_size], data[train_size:], ann_pred, 'ANN')

if __name__ == "__main__":
    main()
