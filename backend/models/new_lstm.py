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

def plot_results(data, train, test, predictions, future_dates, future_predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    
    # Adjust predictions to align with test data indices
    predictions_index = test.index[:len(predictions)]
    plt.plot(predictions_index, predictions, label=f'{model_name} Predictions', color='red')
    
    if future_dates is not None and future_predictions is not None:
        plt.plot(future_dates, future_predictions, label='Future Predictions', color='green')
    
    # Plot the last 120 days of historical prices and the next 30 days of future predictions
    combined_df = pd.concat([data, pd.DataFrame({'Close': future_predictions}, index=future_dates)])
    plt.plot(combined_df.index[-150:], combined_df['Close'][-150:], label='Last 120 Days + 30 Days Prediction')
    
    plt.legend()
    plt.show()

def compute_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

def predict_future(model, last_sequence, scaler, seq_length, horizon_days):
    future_predictions = []
    current_sequence = last_sequence.copy()  # Make a copy to avoid modifying the original
    
    for _ in range(horizon_days):
        prediction_scaled = model.predict(current_sequence.reshape(1, seq_length, 1))
        prediction = scaler.inverse_transform(prediction_scaled).flatten()[0]
        future_predictions.append(prediction)
        
        # Update the sequence with the new prediction
        current_sequence = np.append(current_sequence[1:], prediction_scaled.flatten()).reshape(-1, 1)
    
    return future_predictions

def main():
    np.random.seed(0)
    tf.random.set_seed(0)
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60
    horizon_days = 30  # Number of days to predict into the future

    X, y, scaler = prepare_lstm_data(data, seq_length)
    X_train, y_train = X[:train_size], y[:train_size]  # Use full training set
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]  # Use full test set

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = build_lstm_model(seq_length)

    # Increase training epochs and add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.2, callbacks=[early_stopping])

    lstm_pred = lstm_model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)

    # Scale back the y_test for accuracy computation
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mape = compute_accuracy(y_test_scaled, lstm_pred)
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape * 100}%')

    # Predict future prices
    last_sequence = X_test[-1].copy()  # Make a copy to avoid modifying X_test
    future_predictions = predict_future(lstm_model, last_sequence, scaler, seq_length, horizon_days)
    future_dates = pd.date_range(start=data.index[-1], periods=horizon_days+1)[1:]

    # Plotting results
    plot_results(data, train, test, lstm_pred, future_dates, future_predictions, 'LSTM')

if __name__ == "__main__":
    main()
