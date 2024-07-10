import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Function to load Bitcoin tweets data
def load_tweets_data(file_path):
    print("Loading tweets data...")
    tweets_data = pd.read_csv(file_path)
    print(f"Original tweets data shape: {tweets_data.shape}")

    # Convert created_at to datetime, handling errors
    tweets_data['created_at'] = pd.to_datetime(tweets_data['created_at'], errors='coerce')
    
    # Drop rows with invalid dates
    tweets_data.dropna(subset=['created_at'], inplace=True)

    print(f"Cleaned tweets data shape: {tweets_data.shape}")
    print(f"Sample of cleaned tweets data:\n{tweets_data.head()}")

    tweets_data.set_index('created_at', inplace=True)
    tweets_data = tweets_data.sort_index()

    return tweets_data

# Function to analyze sentiment using VADER
def analyze_sentiment(tweets):
    print("Analyzing sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for i, tweet in enumerate(tweets):
        if i % 100 == 0:
            print(f"Processing tweet {i+1}/{len(tweets)}: {tweet}")
        sentiment = analyzer.polarity_scores(tweet)
        sentiment_scores.append(sentiment['compound'])
        # Debugging: Print sample tweets and their sentiment scores
        if i < 10:  # Increase to check more samples
            print(f"Tweet: {tweet}")
            print(f"Sentiment: {sentiment}")
    print("Sentiment analysis complete.")
    return sentiment_scores

# Function to aggregate sentiment scores over a time window
def aggregate_sentiment_scores(sentiment_scores, tweets_data, time_window='1D'):
    print("Aggregating sentiment scores...")
    df_sentiment = pd.DataFrame(sentiment_scores, columns=['sentiment'], index=tweets_data.index)
    aggregated_sentiment = df_sentiment.resample(time_window).mean().fillna(0)
    print(f"Aggregated sentiment data sample:\n{aggregated_sentiment.head()}")
    print("Aggregation complete.")
    return aggregated_sentiment

# Main function
def main():
    print("Starting main process...")
    tweets_data = load_tweets_data('data\dataset_52-person-from-2021-02-05_2023-06-12_21-34-17-266_with_importance_coefficient_and_clean_text.csv')  # Adjusted to match the file path

    print("Starting sentiment analysis...")
    tweets_data['sentiment_scores'] = analyze_sentiment(tweets_data['full_text'])

    print("Aggregating sentiment scores...")
    sentiment_data = aggregate_sentiment_scores(tweets_data['sentiment_scores'], tweets_data, time_window='1D')

    print("Loading BTC data...")
    btc_data = pd.read_csv('data/BTC-USD.csv')  # Adjusted to match the file path
    print(f"BTC data shape: {btc_data.shape}")
    btc_data['Date'] = pd.to_datetime(btc_data['Date'], errors='coerce')
    btc_data.dropna(subset=['Date'], inplace=True)
    btc_data.set_index('Date', inplace=True)

    print(f"Sample of BTC data:\n{btc_data.head()}")

    print("Merging BTC data with sentiment data...")
    merged_data = pd.merge(btc_data, sentiment_data, left_index=True, right_index=True, how='left')
    merged_data.fillna(0, inplace=True)

    output_file = 'data/BTC-USD_with_sentiment.csv'
    print(f"Saving merged data to {output_file}...")
    merged_data.to_csv(output_file)
    print(f"Sample of merged data:\n{merged_data.head()}")
    print("Process complete.")

if __name__ == "__main__":
    main()
