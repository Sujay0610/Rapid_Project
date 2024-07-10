import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the CSV file
file_path = 'data/tweets_test.csv'
data = pd.read_csv(file_path)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Ensure all values in 'full_text' are strings
data['full_text'] = data['full_text'].fillna('').astype(str)

# Function to get sentiment compound score
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Apply sentiment analysis to the 'full_text' column
data['sentiment_compound'] = data['full_text'].apply(get_sentiment)

# Save the result to a new CSV file
output_file_path = 'data/sentiment_analysis_output.csv'
data.to_csv(output_file_path, index=False)

print(f"Sentiment analysis results saved to {output_file_path}")
