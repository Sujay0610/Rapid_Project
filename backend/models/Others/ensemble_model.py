import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from models.ann_model import load_data, prepare_ann_data, build_ann_model
from models.lstm_model import prepare_lstm_data, build_lstm_model
from models.gru_model import prepare_gru_data, build_gru_model
from models.gbm_model import prepare_gbm_data
from backend.models.Random_forest import prepare_rf_data
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def ensemble_predictions(predictions, weights):
    return np.sum(predictions * weights[:, np.newaxis], axis=0)

def main():
    data = load_data('./data/BTC-USD.csv')
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # ANN predictions
    seq_length = 60
    X_ann, y_ann, scaler_ann = prepare_ann_data(data, seq_length)
    X_train_ann, y_train_ann = X_ann[:train_size-seq_length], y_ann[:train_size-seq_length]
    X_test_ann, y_test_ann = X_ann[train_size-seq_length:], y_ann[train_size-seq_length:]
    ann_model = build_ann_model(seq_length)
    ann_model.fit(X_train_ann, y_train_ann, batch_size=1, epochs=1)
    ann_pred = ann_model.predict(X_test_ann)
    ann_pred = scaler_ann.inverse_transform(ann_pred)

    # LSTM predictions
    X_lstm, y_lstm, scaler_lstm = prepare_lstm_data(data, seq_length)
    X_train_lstm, y_train_lstm = X_lstm[:train_size-seq_length], y_lstm[:train_size-seq_length]
    X_test_lstm, y_test_lstm = X_lstm[train_size-seq_length:], y_lstm[train_size-seq_length:]
    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
    lstm_model = build_lstm_model(seq_length)
    lstm_model.fit(X_train_lstm, y_train_lstm, batch_size=1, epochs=1)
    lstm_pred = lstm_model.predict(X_test_lstm)
    lstm_pred = scaler_lstm.inverse_transform(lstm_pred)

    # GRU predictions
    X_gru, y_gru, scaler_gru = prepare_gru_data(data, seq_length)
    X_train_gru, y_train_gru = X_gru[:train_size-seq_length], y_gru[:train_size-seq_length]
    X_test_gru, y_test_gru = X_gru[train_size-seq_length:], y_gru[train_size-seq_length:]
    X_train_gru = X_train_gru.reshape((X_train_gru.shape[0], X_train_gru.shape[1], 1))
    X_test_gru = X_test_gru.reshape((X_test_gru.shape[0], X_test_gru.shape[1], 1))
    gru_model = build_gru_model(seq_length)
    gru_model.fit(X_train_gru, y_train_gru, batch_size=1, epochs=1)
    gru_pred = gru_model.predict(X_test_gru)
    gru_pred = scaler_gru.inverse_transform(gru_pred)

    # GBM predictions
    X_gbm, y_gbm = prepare_gbm_data(data, seq_length)
    X_train_gbm, y_train_gbm = X_gbm[:train_size-seq_length], y_gbm[:train_size-seq_length]
    X_test_gbm, y_test_gbm = X_gbm[train_size-seq_length:], y_gbm[train_size-seq_length:]
    gbm_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    gbm_model.fit(X_train_gbm, y_train_gbm)
    gbm_pred = gbm_model.predict(X_test_gbm)

    # Random Forest predictions
    X_rf, y_rf = prepare_rf_data(data, seq_length)
    X_train_rf, y_train_rf = X_rf[:train_size-seq_length], y_rf[:train_size-seq_length]
    X_test_rf, y_test_rf = X_rf[train_size-seq_length:], y_rf[train_size-seq_length:]
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train_rf, y_train_rf)
    rf_pred = rf_model.predict(X_test_rf)

    # Ensure the predictions and test set are aligned in length
    test_true = test['Close'][seq_length:]

    # Truncate predictions to match the length of test_true
    min_len = min(len(test_true), len(ann_pred), len(lstm_pred), len(gru_pred), len(gbm_pred), len(rf_pred))
    test_true = test_true[:min_len]
    ann_pred = ann_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    gru_pred = gru_pred[:min_len]
    gbm_pred = gbm_pred[:min_len]
    rf_pred = rf_pred[:min_len]

    # Ensemble predictions
    predictions = np.array([ann_pred.flatten(), lstm_pred.flatten(), gru_pred.flatten(), gbm_pred, rf_pred])
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Example equal weights
    ensemble_pred = ensemble_predictions(predictions, weights)

    # Evaluate the model
    ensemble_mse = mean_squared_error(test_true, ensemble_pred)
    print(f'Ensemble Model MSE: {ensemble_mse}')

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index[:min_len], test_true, label='Actual Prices', color='blue')
    plt.plot(test.index[:min_len], ensemble_pred, label='Ensemble Predictions', color='red')
    plt.legend()
    plt.title('Ensemble Model Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
