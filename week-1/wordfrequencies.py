import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import preprocess_data

# check if twitter_samples exists
try:
    nltk.data.find('corpora/twitter_samples')
except LookupError:
    nltk.download('twitter_samples')

# check if stopwords exists
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# select the lists of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets

# let's see how many tweets we have
print("Number of tweets: ", len(tweets))

labels = np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)

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
    return freqs

freqs = build_freqs(tweets, labels)

print("type of freqs: ", type(freqs))

print("length of freqs: ", len(freqs.keys()))

print("some examples from freqs: \n", list(freqs.items())[:10])

keys = ['happi', 'fuck', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

data = []

for word in keys:
    pos = 0
    neg = 0

    if (word, 1) in freqs:
        pos = freqs[(word, 1)]
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]

    data.append((word, pos, neg))


fig, ax = plt.subplots(figsize=(10, 5))

# convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in data])  

# do the same for the negative counts
y = np.log([x[2] + 1 for x in data]) 

ax.scatter(x, y)

plt.xlabel('Log Positive Count')
plt.ylabel('Log Negative Count')

for i in range(len(data)):
    ax.annotate(data[i][0], (x[i], y[i]))

ax.plot([0, 9], [0, 9], color = "red")  # dashed line y=x
plt.show()