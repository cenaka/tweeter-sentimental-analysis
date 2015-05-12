import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from scipy import sparse, io
import datetime

# FEATURE_EXTRACTORS
import sys
from samples import TRAIN_TWEETS_FILE_NAME, TRAIN_FEATURES_FILE_NAME, TEST_TWEETS_FILE_NAME, TEST_FEATURES_FILE_NAME, \
    POS_TWEETS_FILE_NAME, NEG_TWEETS_FILE_NAME, UNIGRAMMS_TEST_FEATURES_FILE_NAME, UNIGRAMMS_TRAIN_FEATURES_FILE_NAME, \
    TEST_OTHER_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME, STOP_WORDS, BIGRAMMS_TEST_FEATURES_FILE_NAME,\
    BIGRAMMS_TRAIN_FEATURES_FILE_NAME, FOUR_GRAMMS_TEST_FEATURES_FILE_NAME, FOUR_GRAMMS_TRAIN_FEATURES_FILE_NAME

from tweet_prepare import clean_tweet

# FEATURE_EXTRACTORS
NO_WORD_REGEXP = re.compile(r"\bне\b")
URL_REGEXP = re.compile(r"http(s)?://")
BIG_WORDS_REGEXP = re.compile(r"\b[^a-zA-Zа-я\W\s\d]+\b")
REPEAT_LETTERS_REGEXP = re.compile(r"([a-zA-Zа-яА-Я])\1\1|([a-zA-Zа-яА-Я][^\s\d])\2\2|([a-zA-Zа-яА-Я][^\s\d]{2})\3|([a-zA-Zа-яА-Я][^\s\d]{3})\4")
REPEAT_EXCLAMATION_MARK = re.compile(r"!(!+|1{2})")

def has_exclamation_mark(tweet):
    return int('!' in tweet)


def has_repeat_exclamation_mark(tweet):
    return int(bool(REPEAT_EXCLAMATION_MARK.search(tweet)))


def has_question_mark(tweet):
    return int('?' in tweet)


def has_word_not(tweet):
    return int(bool(NO_WORD_REGEXP.search(tweet)))


def has_url(tweet):
    return int(bool(URL_REGEXP.search(tweet)))


def has_big_word(tweet):
    return int(bool(BIG_WORDS_REGEXP.search(tweet)))


def has_repeat_letters(tweet):
    return int(bool(REPEAT_LETTERS_REGEXP.search(tweet)))


def has_mention(tweet):
    return int(bool(MENTION_REGEXP.search(tweet)))

FEATURE_EXTRACTORS = [has_word_not, has_exclamation_mark, has_repeat_exclamation_mark,
                      has_question_mark, has_url, has_big_word, has_repeat_letters, len, has_mention]
# FEATURE EXTRACTION


def extract_features_from_tweet(tweet):
    return np.array([extractor(tweet) for extractor in FEATURE_EXTRACTORS])


def extract_features_from_tweets(tweets, output_file):
    tweet_number = 0
    for tweet in tweets:
        tweet_number += 1
        output_file.write(" ".join(map(str, extract_features_from_tweet(tweet))))
        output_file.write("\n")
        if tweet_number % 100000 == 0:
            print(tweet_number)


def extract_and_dump_features(input_file_path, output_file_path):
    extract_features_from_tweets(open(input_file_path, encoding="utf8"), open(output_file_path, "w"))


def delete_nickname(tweet):
    try:
        return tweet.split("\t")[1]
    except IndexError:
        return tweet


def clean_tweets(tweets):
    cleaned_tweets = []
    for i, tweet in enumerate(tweets.readlines()):
        cleaned_tweets.append(clean_tweet(tweet))
        if i % 10000 == 0:
            print("cleaned {} tweets".format(i))
    return cleaned_tweets


def extract_uni_features_from_tweets(tweets, test_tweets, output_training_file_path, output_test_file_path):
    stop_words = [line.rstrip() for line in open(STOP_WORDS)]
    print(stop_words)
    vectorSK = CountVectorizer(min_df=1, stop_words=stop_words)
    features_matrix = vectorSK.fit_transform(tweets)
    test_matrix = vectorSK.transform(test_tweets)
    print(len(vectorSK.get_feature_names()))
    io.mmwrite(output_training_file_path, features_matrix)
    print("unigrams training matrix dumped")
    io.mmwrite(output_test_file_path, test_matrix)
    print("unigrams test matrix dumped")
    #np.savetxt(output_training_filepath, np.array(features_matrix.toarray()), fmt="%d")
    #np.savetxt(output_test_filepath, np.array(test_matrix.toarray()), fmt="%d")


def extract_bigram_features_from_tweets(tweets, test_tweets, output_training_file_path, output_test_file_path):
    vectorSK = CountVectorizer(ngram_range=(2, 2), min_df=1)
    features_matrix = vectorSK.fit_transform(tweets)
    test_matrix = vectorSK.transform(test_tweets)
    print(len(vectorSK.get_feature_names()))
    io.mmwrite(output_training_file_path, features_matrix)
    print("bigrams training matrix dumped")
    io.mmwrite(output_test_file_path, test_matrix)
    print("bigrams test matrix dumped")


def extract_four_gram_features_from_tweets(tweets, test_tweets, output_training_file_path, output_test_file_path):
    vectorSK = CountVectorizer(ngram_range=(4, 4), analyzer="char", min_df=1)
    features_matrix = vectorSK.fit_transform(tweets)
    test_matrix = vectorSK.transform(test_tweets)
    print(len(vectorSK.get_feature_names()))
    io.mmwrite(output_training_file_path, features_matrix)
    print("four grams training matrix dumped")
    io.mmwrite(output_test_file_path, test_matrix)
    print("four grams test matrix dumped")


def extract_and_dump_ngram_features(input_file_path, input_test_file_path):
    clean_training_tweets = clean_tweets(open(input_file_path))
    clean_test_tweets = clean_tweets(open(input_test_file_path))
    #extract_uni_features_from_tweets(clean_training_tweets, clean_test_tweets,
    #                                 UNIGRAMMS_TRAIN_FEATURES_FILE_NAME, UNIGRAMMS_TEST_FEATURES_FILE_NAME)
    #extract_four_gram_features_from_tweets(clean_training_tweets, clean_test_tweets,
    #                                FOUR_GRAMMS_TRAIN_FEATURES_FILE_NAME, FOUR_GRAMMS_TEST_FEATURES_FILE_NAME)
    extract_bigram_features_from_tweets(clean_training_tweets, clean_test_tweets,
                                        BIGRAMMS_TRAIN_FEATURES_FILE_NAME, BIGRAMMS_TEST_FEATURES_FILE_NAME)


def make_test_and_training_set(pos_tweets, neg_tweets, output_training_file, output_test_file, training_set_size,
                               test_set_size):
    added_tweets = 0
    for tweet in pos_tweets:
        added_tweets += 1
        if training_set_size >= added_tweets:
            output_training_file.write(tweet)
        else:
            if training_set_size + test_set_size >= added_tweets:
                output_test_file.write(tweet)
            else:
                print(added_tweets)
                break
    added_tweets = 0
    for tweet in neg_tweets:
        added_tweets += 1
        if training_set_size >= added_tweets:
            output_training_file.write(tweet)
        else:
            if training_set_size + test_set_size >= added_tweets:
                output_test_file.write(tweet)
            else:
                print(added_tweets)
                break


def make_all_features_matrix(unigrams_file_path, bigrams_file_path, four_grams_file_path,
                             other_features_file_path, all_features_file_path):

    sparse_features_matrix = sparse.csr_matrix(np.loadtxt(other_features_file_path))
    unigrams_matrix = io.mmread(unigrams_file_path)
    bigrams_matrix = io.mmread(bigrams_file_path)
    four_grams_matrix = io.mmread(four_grams_file_path)
    all_features = sp.hstack([unigrams_matrix, bigrams_matrix])
    io.mmwrite(all_features_file_path, all_features)


if __name__ == '__main__':
    make_test_and_training_set(open(POS_TWEETS_FILE_NAME, encoding="utf8"), open(NEG_TWEETS_FILE_NAME, encoding="utf8"),
                      open(TRAIN_TWEETS_FILE_NAME, 'w'), open(TEST_TWEETS_FILE_NAME, 'w'), 4000000, 100000)
    time = datetime.datetime.now()
    extract_and_dump_features(TRAIN_TWEETS_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_features(TEST_TWEETS_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME)
    extract_and_dump_ngram_features(TRAIN_TWEETS_FILE_NAME, TEST_TWEETS_FILE_NAME)
    make_all_features_matrix(UNIGRAMMS_TEST_FEATURES_FILE_NAME, BIGRAMMS_TEST_FEATURES_FILE_NAME,
                             FOUR_GRAMMS_TEST_FEATURES_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME)
    print("test features matrix made")
    make_all_features_matrix(UNIGRAMMS_TRAIN_FEATURES_FILE_NAME, BIGRAMMS_TRAIN_FEATURES_FILE_NAME,
                             FOUR_GRAMMS_TRAIN_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME,
                             TRAIN_FEATURES_FILE_NAME)
    print("training features matrix made")
    print("extract features: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())

# paste other_training_feature_matrix.txt unigrams_training_feature_matrix.txt > all_training_features.txt
# paste other_test_feature_matrix.txt unigrams_test_feature_matrix.txt > all_test_features.txt

