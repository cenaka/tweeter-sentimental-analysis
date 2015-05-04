import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import datetime

# FEATURE_EXTRACTORS
from samples import TRAIN_TWEETS_FILE_NAME, TRAIN_FEATURES_FILE_NAME, TEST_TWEETS_FILE_NAME, TEST_FEATURES_FILE_NAME, \
    POS_TWEETS_FILE_NAME, NEG_TWEETS_FILE_NAME, TEST_UNIGRAMMS_FEATURES_FILE_NAME, TRAIN_UNIGRAMMS_FEATURES_FILE_NAME, \
    TEST_OTHER_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME


def has_exclamation_mark(tweet):
    return '!' in tweet


def has_word_not(tweet):
    pattern = "[^а-яА-Я0-9_]?не[^а-яА-Я0-9_]"
    regexp = re.compile(pattern)
    return len(regexp.findall(tweet))


def has_url(tweet):
    pattern = "http://"
    regexp = re.compile(pattern)
    return len(regexp.findall(tweet)) > 0

FEATURE_EXTRACTORS = [has_word_not, has_exclamation_mark, len]
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


def extract_uni_features_from_tweets(tweets, test_tweets, output_training_filepath, output_test_filepath):
    vectorSK = CountVectorizer(min_df=1, max_features=500)
    features_matrix = vectorSK.fit_transform([delete_nickname(tweet) for tweet in tweets.readlines()])
    test_matrix = vectorSK.transform(test_tweets.readlines())
    for x in vectorSK.get_feature_names():
        print(x)
    # print(test_matrix)
    np.savetxt(output_training_filepath, np.array(features_matrix.toarray()), fmt="%d")
    np.savetxt(output_test_filepath, np.array(test_matrix.toarray()), fmt="%d")


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

if __name__ == '__main__':
    make_test_and_training_set(open(POS_TWEETS_FILE_NAME, encoding="utf8"), open(NEG_TWEETS_FILE_NAME, encoding="utf8"),
                      open(TRAIN_TWEETS_FILE_NAME, 'w'), open(TEST_TWEETS_FILE_NAME, 'w'), 100000)
    time = datetime.datetime.now()
    extract_and_dump_features(TRAIN_TWEETS_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_features(TEST_TWEETS_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_unigram_features(TRAIN_TWEETS_FILE_NAME, TRAIN_UNIGRAMMS_FEATURES_FILE_NAME, TEST_TWEETS_FILE_NAME,
                                      TEST_UNIGRAMMS_FEATURES_FILE_NAME)
    print("extract features: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())

#paste other_training_feature_matrix.txt unigrams_training_feature_matrix.txt > feature_matrix.txt
#

