import tweepy
import nltk
from nltk.corpus import stopwords
import re
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

consumer_key = os.getenv('consumer_key')
consumer_secret = os.getenv('consumer_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Collect tweets with a specific hashtag
tweets = api.search(q="#examplehashtag", lang="en", count=100)
tweet_texts = [tweet.text for tweet in tweets]

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return text

cleaned_tweets = [clean_text(tweet) for tweet in tweet_texts]

sentiment_model = pipeline("sentiment-analysis")
sentiments = sentiment_model(cleaned_tweets)

df = pd.DataFrame(sentiments, columns=['Sentiment'])
sentiment_counts = df['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar')
plt.show()