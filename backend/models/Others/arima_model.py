import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

def train_arima_model(train):
    auto_arima_model = auto_arima(train['Close'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    arima_order = auto_arima_model.order
    arima_model = sm.tsa.ARIMA(train['Close'], order=arima_order)
    arima_fit = arima_model.fit()
    return arima_fit

def predict_arima_model(arima_fit, train, test):
    arima_pred = arima_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
    return arima_pred

def plot_results(train, test, predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    plt.plot(test.index, predictions, label=f'{model_name} Predictions', color='red')
    plt.legend()
    plt.show()

def main():
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    arima_fit = train_arima_model(train)
    arima_pred = predict_arima_model(arima_fit, train, test)
    arima_mse = mean_squared_error(test['Close'], arima_pred)
    print(f'ARIMA Model MSE: {arima_mse}')

    plot_results(train, test, arima_pred, 'ARIMA')

if __name__ == "__main__":
    main()