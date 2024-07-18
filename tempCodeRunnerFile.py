from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import io
import base64

app = Flask(__name__)

# Function to load data
def load_data(file):
    data = pd.read_csv(file)
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
# Function to prepare data for ANN and LSTM
def prepare_ann_lstm_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def prepare_rf_data(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data['Close'].iloc[i-seq_length:i].values)
        y.append(data['Close'].iloc[i])
    return np.array(X), np.array(y)

def prepare_gbm_data(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data['Close'].iloc[i-seq_length:i].values)
        y.append(data['Close'].iloc[i])
    return np.array(X), np.array(y)
# Function to prepare data for GBM and RFR
def prepare_gbm_rf_data(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data['Close'].iloc[i-seq_length:i].values)
        y.append(data['Close'].iloc[i])
    return np.array(X), np.array(y)

# Function to build ANN model
def build_ann_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build LSTM model
def build_lstm_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to build GRU model
def build_gru_model(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.GRU(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to plot results
def plot_results(train, test, predictions, model_name, seq_length):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
    plt.plot(test.index[seq_length:], predictions, label=f'{model_name} Predictions', color='red')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    crypto = request.form['crypto']
    model_choice = request.form['model']
    file = request.files['csvFile']
    seq_length = int(request.form['seq_length'])

    # Load data based on cryptocurrency choice
    data = load_data(file)

    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    if model_choice == 'ANN':
        X, y, scaler = prepare_ann_data(data, seq_length)
        X_train, y_train = X[:train_size-seq_length], y[:train_size-seq_length]
        X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]

        ann_model = build_ann_model(seq_length)
        ann_model.fit(X_train, y_train, batch_size=1, epochs=5)

        ann_pred_scaled = ann_model.predict(X_test)
        ann_pred = scaler.inverse_transform(ann_pred_scaled).flatten()

        test_true = data['Close'].values[train_size:]

        if len(ann_pred) != len(test_true):
            print("Warning: Lengths of ann_pred and test_true do not match!")
            min_len = min(len(ann_pred), len(test_true))
            ann_pred = ann_pred[:min_len]
            test_true = test_true[:min_len]
        
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train['Close'], label='Training Data')
        plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
        plt.plot(test.index, ann_pred, label='ANN Predictions', color='red')
        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
            
        
        def calculate_mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true))       
        
        mse = np.sqrt(mean_squared_error(test_true, ann_pred))
        mape = calculate_mape(test_true, ann_pred)
       
        return jsonify({'mse': mse, 'mape': mape, 'plot_url': plot_url})
    
    elif model_choice == 'LSTM':
        X, y, scaler = prepare_ann_lstm_data(data, seq_length)
        X_train, y_train = X[:train_size], y[:train_size]  # Use full training set
        X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]  # Use full test set

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        lstm_model = build_lstm_model(seq_length)
        lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

        lstm_pred = lstm_model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        def plot_results(train, test, predictions, model_name):
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train['Close'], label='Training Data')
            plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
            
            # Adjust predictions to align with test data indices
            predictions_index = test.index[:len(predictions)]
            plt.plot(predictions_index, predictions, label=f'{model_name} Predictions', color='red')
            
            plt.legend()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            return plot_url
        def compute_accuracy(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred)
            return rmse, mape
                # Scale back the y_test for accuracy computation
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse, mape = compute_accuracy(y_test_scaled, lstm_pred)
        plot_url= plot_results(train, test, lstm_pred, 'LSTM')
        return jsonify({'mse': rmse, 'mape': mape, 'plot_url': plot_url})

    elif model_choice == 'LSTM-GRU':
        X, y, scaler = prepare_ann_lstm_data(data, seq_length)
        X_train, y_train = X[:train_size], y[:train_size]  # Use full training set
        X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]  # Use full test set
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        lstm_model = build_lstm_model(seq_length)
        lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)
        gru_model = build_gru_model(seq_length)
        gru_model.fit(X_train, y_train, batch_size=1, epochs=1)
        lstm_pred = lstm_model.predict(X_test)
        gru_pred = gru_model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        gru_pred = scaler.inverse_transform(gru_pred)
        alpha = 0.5  # Weight parameter for exponential formula
        ensemble_pred = alpha * lstm_pred + (1 - alpha) * gru_pred
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        def compute_accuracy(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred)
            return rmse, mape
        def plot_results(train, test, predictions, model_name):
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train['Close'], label='Training Data')
            plt.plot(test.index, test['Close'], label='Actual Prices', color='blue')
            
            # Adjust predictions to align with test data indices
            predictions_index = test.index[:len(predictions)]
            plt.plot(predictions_index, predictions, label=f'{model_name} Predictions', color='red')
            
            plt.legend()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            return plot_url
        rmse, mape = compute_accuracy(y_test_scaled, ensemble_pred)
        plot_url= plot_results(train, test, ensemble_pred, 'Ensemble Without Sentiment Predictions')
        return jsonify({'mse': rmse, 'mape': mape, 'plot_url': plot_url})
    
    elif model_choice == 'GBM':
        X, y = prepare_gbm_data(data, seq_length)
        X_train, y_train = X[:train_size-seq_length], y[:train_size-seq_length]
        X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]

        gbm_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        gbm_model.fit(X_train, y_train)
        gbm_pred = gbm_model.predict(X_test)

        mse = np.sqrt(mean_squared_error(y_test, gbm_pred))
        mape = mean_absolute_percentage_error(y_test, gbm_pred)

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Actual Prices')
        plt.plot(data.index[train_size:], gbm_pred, label='GBM Predictions', color='red')
        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return jsonify({'mse': mse, 'mape': mape, 'plot_url': plot_url})

    elif model_choice == 'RFR':
        X, y = prepare_rf_data(data, seq_length)
        X_train, y_train = X[:train_size-seq_length], y[:train_size-seq_length]
        X_test, y_test = X[train_size-seq_length:], y[train_size-seq_length:]

        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Actual Prices')
        plt.plot(data.index[train_size:], rf_pred, label='Random Forest Predictions', color='red')
        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return jsonify({'mse': rf_mse, 'mape': rf_mape, 'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
