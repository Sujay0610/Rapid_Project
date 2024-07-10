import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
    data.set_index('Date', inplace=True)
    data = data.fillna(method='ffill')
    return data

# Function to fine-tune GPT-2 on historical data
def fine_tune_gpt(data):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2Model.from_pretrained('gpt2')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    for text, target_price in zip(data['MarketSentiment'], data['Close']):
        inputs = tokenizer(text, return_tensors="tf")
        outputs = model(inputs)
        loss = tf.keras.losses.mean_squared_error(target_price, outputs)
        grads = tf.gradients(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Function to predict using fine-tuned GPT-2
def predict_gpt(data):
    predictions = []
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2Model.from_pretrained('gpt2')
    
    for text in data['MarketSentiment']:
        inputs = tokenizer(text, return_tensors="tf")
        outputs = model(inputs)
        predictions.append(outputs)

    return predictions

# Main function
def main():
    data = load_data('data/BTC-USD.csv')  # Ensure 'MarketSentiment' column exists in your CSV
    fine_tune_gpt(data)
    predictions = predict_gpt(data)

    true_prices = data['Close']
    mse = mean_squared_error(true_prices, predictions)
    print(f'MSE: {mse}')

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, true_prices, label='Actual Prices')
    plt.plot(data.index, predictions, label='Predicted Prices', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
