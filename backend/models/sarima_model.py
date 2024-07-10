import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # SARIMA Model
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    sarima_model = sm.tsa.statespace.SARIMAX(train['Close'], order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)

    # Forecast
    sarima_pred = sarima_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')

    # Calculate MSE and MAPE
    sarima_mse = mean_squared_error(test['Close'], sarima_pred)
    sarima_mape = calculate_mape(test['Close'], sarima_pred)
    
    print(f'SARIMA Model MSE: {sarima_mse}')
    print(f'SARIMA Model MAPE: {sarima_mape}%')

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    plt.plot(test.index, sarima_pred, label='SARIMA Predictions', color='red')
    plt.legend()
    plt.show()
