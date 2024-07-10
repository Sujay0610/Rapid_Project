#rfr model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

def prepare_rf_data(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data['Close'].iloc[i-seq_length:i].values)
        y.append(data['Close'].iloc[i])
    return np.array(X), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60
    X, y = prepare_rf_data(data, seq_length)
    X_train, y_train = X[:train_size-seq_length], y[:train_size-seq_length]
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
    print(f'Random Forest Model MSE: {rf_mse}')
    print(f'Random Forest Model MAPE: {rf_mape}%')

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Prices')
    plt.plot(data.index[train_size:], rf_pred, label='Random Forest Predictions', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
