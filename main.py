import tweepy
import nltk
from nltk.corpus import stopwords
import re
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import time

load_dotenv()

# get environment keys from .env
consumer_key = os.getenv('consumer_key')
consumer_secret = os.getenv('consumer_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')
bearer_token = os.getenv('bearer_token')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

client = tweepy.Client(bearer_token=bearer_token)

# Set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# change model based on desires
# this is a compact, social media tuned model which is perfect for what I want
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device="cpu")


# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return text

# Function to collect tweets with a specific hashtag
def get_tweets(query):
    tweets = []
    try:
        response = client.search_recent_tweets(query=query, max_results=10)
        if response.data:
            tweets.extend([tweet.text for tweet in response.data])
        else:
            print("No tweets found.")
    except tweepy.errors.TooManyRequests:
        print("Rate limit exceeded. Waiting 15 minutes...")
        time.sleep(15 * 60)  # Wait 15 minutes before retrying
    return tweets

# Function to analyze sentiment of text
def analyze_sentiment(texts):
    cleaned_texts = [clean_text(text) for text in texts]
    sentiments = sentiment_model(cleaned_texts)
    df = pd.DataFrame(sentiments, columns=['label', 'score'])
    sentiment_counts = df['label'].value_counts()
    sentiment_counts.plot(kind='bar')
    plt.show()
    return sentiments

def display_sentiment_result(result):
    label = result[0]['label']
    score = result[0]['score']
    print(f"Predicted Sentiment: {label} (confidence: {score:.2f})\n")

# Function to analyze sentiment of a custom paragraph
def analyze_paragraph(paragraph):
    cleaned_paragraph = clean_text(paragraph)
    result = sentiment_model(cleaned_paragraph)
    display_sentiment_result(result)

# Main function to handle modes
def main():
    mode = input("Choose a mode: 'interactive' for text input or 'twitter' for Twitter analysis: ").strip().lower()
    
    if mode == 'interactive':
        paragraph = input("Enter a paragraph to analyze its sentiment: ")
        analyze_paragraph(paragraph)
        
    elif mode == 'twitter':
        query = input("Enter a hashtag to analyze tweets (e.g., #example): ")
        tweet_texts = get_tweets(f"{query} lang:en")
        if tweet_texts:
            analyze_sentiment(tweet_texts)
    else:
        print("Invalid mode. Please choose 'interactive' or 'twitter'.")

# Run the main function
if __name__ == "__main__":
    main()