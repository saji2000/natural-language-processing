import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

def preprocess_data(words):
    """
    Preprocess a given tweet string.
    
    This function takes a tweet string and preprocesses it by removing special characters, 
    hyperlinks, hashtags, punctuation, and stop words, and then stemming the remaining words.
    
    Input:
        words (str): a tweet string
    Output:
        words_clean (list): a list of cleaned and stemmed words
    """
    words2 = re.sub(r'^RT[\s]+', '', words)

    # remove hyperlinks
    words2 = re.sub(r'https?://[^\s\n\r]+', '', words2)

    # remove hashtags
    # only removing the hash # sign from the word
    words2 = re.sub(r'#', '', words2)

    # tokenize the tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    words2 = tokenizer.tokenize(words2)

    # remove stop words
    words2 = [word for word in words2 if word not in stopwords.words('english')]

    # stemming
    stemmer = PorterStemmer()
    words_clean = [stemmer.stem(word) for word in words2]

    return words_clean