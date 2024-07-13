import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm  # Import tqdm for progress bar

# Download NLTK data files
nltk.download('wordnet')
nltk.download('stopwords')

# Load the CSV file
file_path = 'data/tweets_test.csv'
data = pd.read_csv(file_path)

# Initialize VADER sentiment analyzer and BERT model
analyzer = SentimentIntensityAnalyzer()
bert_pipeline = pipeline("sentiment-analysis")

# Ensure all values in 'full_text' are strings
data['full_text'] = data['full_text'].fillna('').astype(str)

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
    return text

# Function to get sentiment compound score
def get_vader_sentiment(text):
    cleaned_text = clean_text(text)
    sentiment = analyzer.polarity_scores(cleaned_text)
    return sentiment['compound']

# Function to get BERT sentiment score
def get_bert_sentiment(text):
    cleaned_text = clean_text(text)
    sentiment = bert_pipeline(cleaned_text)
    return sentiment[0]['score'] if sentiment[0]['label'] == 'POSITIVE' else -sentiment[0]['score']

# Apply sentiment analysis to the 'full_text' column with progress bar
tqdm.pandas()  # Initialize tqdm with pandas
data['vader_sentiment'] = data['full_text'].progress_apply(get_vader_sentiment)
data['bert_sentiment'] = data['full_text'].progress_apply(get_bert_sentiment)

# Combine the sentiment scores (simple average in this case)
data['sentiment_compound'] = (data['vader_sentiment'] + data['bert_sentiment']) / 2

# Save the result to a new CSV file
output_file_path = 'data/sentiment_analysis_output_advanced.csv'
data.to_csv(output_file_path, index=False)

print(f"Sentiment analysis results saved to {output_file_path}")
