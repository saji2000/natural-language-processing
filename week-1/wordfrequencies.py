import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import preprocess_data

nltk.download('twitter_samples')

nltk.download('stopwords')

# select the lists of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets

# let's see how many tweets we have
print("Number of tweets: ", len(tweets))

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()

    freqs = {}

    for y, tweet in zip(yslist, tweets):
        for word in preprocess_data(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1