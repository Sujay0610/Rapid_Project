import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import timedelta

# Function to load data
def load_data(price_file_path):
    price_data = pd.read_csv(price_file_path)
    
    price_data['Date'] = pd.to_datetime(price_data['Date'], format='%m/%d/%y')
    price_data.set_index('Date', inplace=True)
    price_data = price_data.fillna(method='ffill')
    
    return price_data

# Function to prepare data for LSTM/GRU
def prepare_lstm_data(data, seq_length):
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
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build GRU model
def build_gru_model(seq_length, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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

# Function to predict future prices using rolling predictions
def predict_future(data, lstm_model, gru_model, scaler, seq_length, num_features, future_steps):
    recent_data = data[-seq_length:]
    future_predictions = []

    for _ in range(future_steps):
        scaled_recent_data = scaler.transform(recent_data)
        scaled_recent_data = scaled_recent_data.reshape((1, seq_length, num_features))
        
        lstm_pred = lstm_model.predict(scaled_recent_data)[0, 0]
        gru_pred = gru_model.predict(scaled_recent_data)[0, 0]
        
        # Ensemble prediction
        prediction = 0.5 * lstm_pred + 0.5 * gru_pred
        future_predictions.append(prediction)
        
        new_row = np.append(prediction, np.zeros(num_features - 1))
        new_row = scaler.inverse_transform(new_row.reshape(1, -1))[0]
        
        recent_data = np.append(recent_data[1:], new_row.reshape(1, -1), axis=0)
    
    future_predictions = scaler.inverse_transform(np.hstack((np.array(future_predictions).reshape(-1, 1), np.zeros((len(future_predictions), num_features - 1)))))[:, 0]
    return future_predictions

# Main function
def main():
    # Load data
    data = load_data('data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    seq_length = 60

    # Prepare data
    X, y, scaler = prepare_lstm_data(data, seq_length)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]
    
    num_features = X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

    # Build LSTM model
    lstm_model = build_lstm_model(seq_length, num_features)

    # Define early stopping for LSTM
    early_stopping_lstm = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train LSTM model
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, callbacks=[early_stopping_lstm])

    # Build GRU model
    gru_model = build_gru_model(seq_length, num_features)

    # Define early stopping for GRU
    early_stopping_gru = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train GRU model
    gru_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, callbacks=[early_stopping_gru])

    # Predictions from LSTM and GRU models
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)

    # Inverse transform predictions to original scale
    lstm_pred = scaler.inverse_transform(np.hstack((lstm_pred, np.zeros((lstm_pred.shape[0], num_features - 1)))))[:, 0]
    gru_pred = scaler.inverse_transform(np.hstack((gru_pred, np.zeros((gru_pred.shape[0], num_features - 1)))))[:, 0]

    # Ensemble predictions
    ensemble_pred = 0.5 * lstm_pred + 0.5 * gru_pred

    # Compute accuracy metrics for ensemble
    y_test_scaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1)))))[:, 0]
    rmse, mape = compute_accuracy(y_test_scaled, ensemble_pred)
    print(f'Ensemble RMSE: {rmse}')
    print(f'Ensemble MAPE: {mape * 100}%')

    plot_results(train, test, ensemble_pred, 'LSTM-GRU Ensemble Prediction')

    # Future prediction using rolling predictions
    future_steps = 30  # Predict the next 30 days
    future_predictions_ensemble = predict_future(data.values, lstm_model, gru_model, scaler, seq_length, num_features, future_steps)

    # Create a DataFrame for future predictions
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Future_Ensemble_Predictions': future_predictions_ensemble
    })
    future_df.set_index('Date', inplace=True)

    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Prices', color='blue')
    plt.plot(future_df.index, future_df['Future_Ensemble_Predictions'], label='Future Ensemble Predictions', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
