import numpy as np

# Exponential Moving Average (EMA) function for LLM (Llama-3) predictions
def llama3_predict(data, alpha=0.5):
    predictions = [data.iloc[0]]  # Start with the first value as initial prediction
    for i in range(1, len(data)):
        prediction = alpha * data.iloc[i] + (1 - alpha) * predictions[-1]
        predictions.append(prediction)
    return np.array(predictions)

def main():
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    data = pd.read_csv('data/BTC-USD.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')

    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    llm_pred = llama3_predict(train['Close'], alpha=0.2)  # Using EMA on training data for simplicity
    llm_mse = mean_squared_error(test['Close'], llm_pred[-len(test):])
    print(f'LLM (Llama-3) Model MSE: {llm_mse}')

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    plt.plot(test.index, llm_pred[-len(test):], label='LLM Predictions (EMA)', color='red')  # Plot only predictions for test data
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
