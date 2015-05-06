import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from scipy import sparse, io
import datetime

# FEATURE_EXTRACTORS
import sys
from samples import TRAIN_TWEETS_FILE_NAME, TRAIN_FEATURES_FILE_NAME, TEST_TWEETS_FILE_NAME, TEST_FEATURES_FILE_NAME, \
    POS_TWEETS_FILE_NAME, NEG_TWEETS_FILE_NAME, TEST_UNIGRAMMS_FEATURES_FILE_NAME, TRAIN_UNIGRAMMS_FEATURES_FILE_NAME, \
    TEST_OTHER_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME, STOP_WORDS

from tweet_prepare import clean_tweet

NO_WORD_REGEXP = re.compile(r"\bне\b")
URL_REGEXP = re.compile(r"http(s)?://")

def has_exclamation_mark(tweet):
    return int('!' in tweet)

def has_word_not(tweet):
    return int(bool(NO_WORD_REGEXP.search(tweet)))

def has_url(tweet):
    return int(bool(URL_REGEXP.search(tweet)))

FEATURE_EXTRACTORS = [has_word_not, has_exclamation_mark, has_url, len]
# FEATURE EXTRACTION


def extract_features_from_tweet(tweet):
    return np.array([extractor(tweet) for extractor in FEATURE_EXTRACTORS])


def extract_features_from_tweets(tweets, output_file):
    tweet_number = 0
    for tweet in tweets:
        tweet_number += 1
        output_file.write(" ".join(map(str, extract_features_from_tweet(tweet))))
        output_file.write("\n")
        if tweet_number % 10000 == 0:
            print(tweet_number, tweet)


def delete_nickname(tweet):
    try:
        return tweet.split("\t")[1]
    except IndexError:
        return tweet


def clean_tweets(tweets):
    cleaned_tweets = []
    for i, tweet in enumerate(tweets.readlines()):
        cleaned_tweets.append(clean_tweet(tweet))
        if i % 1000 == 0:
            print("cleaned {} tweets".format(i))
    return cleaned_tweets


def extract_uni_features_from_tweets(tweets, test_tweets, output_training_filepath, output_test_filepath):
    stop_words = [line.rstrip() for line in open(STOP_WORDS)]
    print(stop_words)
    vectorSK = CountVectorizer(min_df=1, max_features=5000, stop_words=stop_words)
    features_matrix = vectorSK.fit_transform(clean_tweets(tweets))
    test_matrix = vectorSK.transform(clean_tweets(test_tweets))
    for x in vectorSK.get_feature_names():
        print(x)
    io.mmwrite(output_training_filepath, features_matrix)
    print("unigrams training matrix dumped")
    io.mmwrite(output_test_filepath, test_matrix)
    print("unigrams test matrix dumped")
    #np.savetxt(output_training_filepath, np.array(features_matrix.toarray()), fmt="%d")
    #np.savetxt(output_test_filepath, np.array(test_matrix.toarray()), fmt="%d")


def extract_and_dump_features(input_filepath, output_filepath):
    extract_features_from_tweets(open(input_filepath, encoding="utf8"), open(output_filepath, "w"))


def extract_and_dump_unigram_features(input_filepath, output_filepath, input_test_filepath, output_test_filepath):
    extract_uni_features_from_tweets(
        open(input_filepath, encoding="utf8"), open(input_test_filepath, encoding="utf8"),
        output_filepath, output_test_filepath)


def make_test_and_training_set(pos_tweets, neg_tweets, output_training_file, output_test_file, number):
    added_tweets = 0
    for tweet in pos_tweets:
        added_tweets += 1
        if number >= added_tweets:
            output_training_file.write(tweet)
        else:
            if 2 * number >= added_tweets:
                output_test_file.write(tweet)
            else:
                break
    added_tweets = 0
    for tweet in neg_tweets:
        added_tweets += 1
        if number >= added_tweets:
            output_training_file.write(tweet)
        else:
            if 2 * number >= added_tweets:
                output_test_file.write(tweet)
            else:
                break


def make_all_features_matrix(unigrams_filepath, other_features_filepath, all_features_file_path):
    sparse_features_matrix = sparse.csr_matrix(np.loadtxt(other_features_filepath))
    unigrams_matrix = io.mmread(unigrams_filepath)
    all_features = sp.hstack([sparse_features_matrix, unigrams_matrix])
    io.mmwrite(all_features_file_path, all_features)


if __name__ == '__main__':
    make_test_and_training_set(open(POS_TWEETS_FILE_NAME, encoding="utf8"), open(NEG_TWEETS_FILE_NAME, encoding="utf8"),
                      open(TRAIN_TWEETS_FILE_NAME, 'w'), open(TEST_TWEETS_FILE_NAME, 'w'), 100000)
    time = datetime.datetime.now()
    extract_and_dump_features(TRAIN_TWEETS_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_features(TEST_TWEETS_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_unigram_features(TRAIN_TWEETS_FILE_NAME, TRAIN_UNIGRAMMS_FEATURES_FILE_NAME, TEST_TWEETS_FILE_NAME,
                                      TEST_UNIGRAMMS_FEATURES_FILE_NAME)
    make_all_features_matrix(TEST_UNIGRAMMS_FEATURES_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME)
    make_all_features_matrix(TRAIN_UNIGRAMMS_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME, TRAIN_FEATURES_FILE_NAME)

    print("extract features: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())

# paste other_training_feature_matrix.txt unigrams_training_feature_matrix.txt > all_training_features.txt
# paste other_test_feature_matrix.txt unigrams_test_feature_matrix.txt > all_test_features.txt

