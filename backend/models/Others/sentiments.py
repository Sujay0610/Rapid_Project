import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Function to load Bitcoin tweets data
def load_tweets_data(file_path):
    tweets_data = pd.read_csv(file_path)
    tweets_data['user_created'] = pd.to_datetime(tweets_data['user_created'])
    tweets_data.set_index('user_created', inplace=True)
    tweets_data = tweets_data.sort_index()
    return tweets_data

# Function to analyze sentiment using VADER
def analyze_sentiment(tweets):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for tweet in tweets:
        sentiment = analyzer.polarity_scores(tweet)
        sentiment_scores.append(sentiment)
    return sentiment_scores

# Function to aggregate sentiment scores over a time window
def aggregate_sentiment_scores(sentiment_scores, time_window='1D'):
    df_sentiment = pd.DataFrame(sentiment_scores)
    df_sentiment.index = pd.to_datetime(df_sentiment.index)
    aggregated_sentiment = df_sentiment.resample(time_window).mean().fillna(0)
    return aggregated_sentiment

# Function to prepare data for LSTM
def prepare_lstm_data(data, sentiment_data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])
    
    merged_data = pd.merge_asof(data, sentiment_data, left_index=True, right_index=True)
    merged_data.fillna(0, inplace=True)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(np.concatenate((scaled_data[i-seq_length:i, 0], merged_data.iloc[i-seq_length:i, 1:])))
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# Function to prepare data for GRU
def prepare_gru_data(data, sentiment_data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])
    
    merged_data = pd.merge_asof(data, sentiment_data, left_index=True, right_index=True)
    merged_data.fillna(0, inplace=True)
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(np.concatenate((scaled_data[i-seq_length:i, 0], merged_data.iloc[i-seq_length:i, 1:])))
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

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
def plot_results(train, test, predictions_lstm, predictions_gru, predictions_ensemble):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['close'], label='Training Data')
    plt.plot(test.index, test['close'], label='Actual Prices', color='blue')
    
    predictions_index = test.index[:len(predictions_lstm)]
    plt.plot(predictions_index, predictions_lstm, label='LSTM Predictions', color='red', linestyle='--')
    plt.plot(predictions_index, predictions_gru, label='GRU Predictions', color='green', linestyle='--')
    plt.plot(predictions_index, predictions_ensemble, label='Ensemble Predictions', color='purple')
    
    plt.legend()
    plt.show()

# Function to compute accuracy metrics
def compute_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape

# Function to blend predictions using exponential decay
def blend_predictions(predictions_lstm, predictions_gru, alpha):
    weights_lstm = np.exp(alpha * np.arange(len(predictions_lstm))[::-1])
    weights_gru = np.exp(alpha * np.arange(len(predictions_gru))[::-1])
    
    weights_lstm /= np.sum(weights_lstm)
    weights_gru /= np.sum(weights_gru)
    
    blended_predictions = weights_lstm * predictions_lstm + weights_gru * predictions_gru
    
    return blended_predictions

# Main function
def main():
    data = pd.read_csv('data/BTC-USD.csv')
    data['date'] = pd.to_datetime(data['Date'])
    data.set_index('date', inplace=True)
    data = data.fillna(method='ffill')
    
    tweets_data = load_tweets_data('data/bitcoin_tweets.csv')
    
    tweets_data['sentiment_scores'] = analyze_sentiment(tweets_data['tweet'])
    
    sentiment_data = aggregate_sentiment_scores(tweets_data['sentiment_scores'], time_window='1D')

    seq_length = 60
    X_lstm, y_lstm, scaler_lstm = prepare_lstm_data(data, sentiment_data, seq_length)
    X_gru, y_gru, scaler_gru = prepare_gru_data(data, sentiment_data, seq_length)
    
    train_size = int(len(data) * 0.8)
    X_train_lstm, y_train_lstm = X_lstm[:train_size]
    X_test_lstm, y_test_lstm = X_lstm[train_size-seq_length:]
    
    X_train_gru, y_train_gru = X_gru[:train_size]
    X_test_gru, y_test_gru = X_gru[train_size-seq_length:]

    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
    
    X_train_gru = X_train_gru.reshape((X_train_gru.shape[0], X_train_gru.shape[1], 1))
    X_test_gru = X_test_gru.reshape((X_test_gru.shape[0], X_test_gru.shape[1], 1))

    lstm_model = build_lstm_model(seq_length)
    lstm_model.fit(X_train_lstm, y_train_lstm, batch_size=1, epochs=1, verbose=2)

    gru_model = build_gru_model(seq_length)
    gru_model.fit(X_train_gru, y_train_gru, batch_size=1, epochs=1, verbose=2)

    lstm_pred = lstm_model.predict(X_test_lstm)
    gru_pred = gru_model.predict(X_test_gru)

    lstm_pred = scaler_lstm.inverse_transform(lstm_pred)
    gru_pred = scaler_gru.inverse_transform(gru_pred)

    y_test_scaled_lstm = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1))
    y_test_scaled_gru = scaler_gru.inverse_transform(y_test_gru.reshape(-1, 1))

    lstm_rmse, lstm_mape = compute_accuracy(y_test_scaled_lstm, lstm_pred)
    print(f'LSTM RMSE: {lstm_rmse}')
    print(f'LSTM MAPE: {lstm_mape * 100}%')

    gru_rmse, gru_mape = compute_accuracy(y_test_scaled_gru, gru_pred)
    print(f'GRU RMSE: {gru_rmse}')
    print(f'GRU MAPE: {gru_mape * 100}%')

    alpha = 0.05
    blended_predictions = blend_predictions(lstm_pred.flatten(), gru_pred.flatten(), alpha)

    ensemble_rmse, ensemble_mape = compute_accuracy(y_test_scaled_lstm, blended_predictions)
    print(f'Ensemble (LSTM + GRU) RMSE: {ensemble_rmse}')
    print(f'Ensemble (LSTM + GRU) MAPE: {ensemble_mape * 100}%')

    plot_results(data.iloc[:train_size], data.iloc[train_size:], lstm_pred, gru_pred, blended_predictions)

if __name__ == "__main__":
    main()
