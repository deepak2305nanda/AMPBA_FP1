import numpy as np
import pandas as pd
import cloudpickle
import requests
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import yfinance as yf


# Loading the trained machine learning model
def load_model_from_github():
    model_url = "https://raw.githubusercontent.com/deepak2305nanda/AMPBA_FP1/main/model.sav"
    response = requests.get(model_url)
    model_bytes = BytesIO(response.content)
    return cloudpickle.load(model_bytes)

model = load_model_from_github()

# Mapping dictionary for sentiment labels
sentiment_mapping = {0: 'Bearish', 1: 'Bullish'}

# Text preprocessing function
def preprocess_text(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    tweet = ' '.join([word for word in word_tokens if word not in stop_words])
    return tweet

# Predicting sentiment function
def predict_stock_sentiment(tweet):
    tweet = preprocess_text(tweet)
    # Predict sentiment using the loaded model
    prediction = model.predict([tweet])
    return prediction[0]

# Mapping sentiment to labels function
def map_sentiment(sentiment):
    return sentiment_mapping.get(sentiment, 'Unknown')

# Function to fetch stock data using yfinance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to plot moving averages
def plot_moving_averages(stock_data, ma1, ma2):
    fig, ax = plt.subplots()
    
    # Plotting Closing Prices
    ax.plot(stock_data['Close'], label='Closing Price', color='blue')

    # Calculating and plotting Moving Averages
    ax.plot(stock_data['Close'].rolling(window=ma1).mean(), label=f'MA {ma1} Days', color='green')
    ax.plot(stock_data['Close'].rolling(window=ma2).mean(), label=f'MA {ma2} Days', color='red')

    # Adding labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('ICICI Stock Prices with Moving Averages')
    ax.legend()

    return fig

# Main Streamlit app function
def main():
    st.title("ICICI Stock Sentiment Predictor - Group-10 AMPBA Co'24 Summer")
    
    # HTML styling for the app
    html_temp = """
    <div style="background-color: tomato; padding: 10px;">
    <h2 style="color: white; text-align: center;">Streamlit Stock Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # User input for the tweet
    Stock_Reviews = st.text_input("Tweet", "Type here")
    result = ""

    # Predict button click event
    if st.button("Predict"):
        # Predict sentiment and map to label
        sentiment = predict_stock_sentiment(Stock_Reviews)
        result = map_sentiment(sentiment)

    # Displaying the predicted sentiment
    st.success('The sentiment is {}'.format(result))

    # User input for desired timeframe
    start_date = st.date_input("Select start date", pd.to_datetime("2012-01-01"))
    end_date = st.date_input("Select end date", pd.to_datetime("2022-01-01"))

    # Fetching stock data
    icici_stock_data = get_stock_data("ICICIBANK.BO", start_date, end_date)

    # Displaying the chart with moving averages
    st.subheader("ICICI Stock Chart with Moving Averages")
    ma1 = st.slider("Select Moving Average (MA) for 100 Days", min_value=1, max_value=1000, value=100)
    ma2 = st.slider("Select Moving Average (MA) for 200 Days", min_value=1, max_value=1000, value=200)

    # Plotting and displaying the chart
    fig = plot_moving_averages(icici_stock_data, ma1, ma2)
    st.pyplot(fig)

# Running the app
if __name__ == '__main__':
    main()
