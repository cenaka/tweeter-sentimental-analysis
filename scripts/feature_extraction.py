import numpy as np

# FEATURE_EXTRACTORS
from samples import TRAIN_TWEETS_FILE_NAME, TRAIN_FEATURES_FILE_NAME, TEST_TWEETS_FILE_NAME, TEST_FEATURES_FILE_NAME


def has_positive_smile(tweet):
    positive_smiles = [':)', ':-)', ': )', ':D', '=)', '))']
    return int(any((positive_smile in tweet for positive_smile in positive_smiles)))

FEATURE_EXTRACTORS = [has_positive_smile, len]  # just an example features

# FEATURE EXTRACTION

def extract_features_from_tweet(tweet):
    return np.array([extractor(tweet) for extractor in FEATURE_EXTRACTORS])


def extract_features_from_tweets(tweets):
    features_matrix = np.empty((0, len(FEATURE_EXTRACTORS)))
    for tweet in tweets:
        features_matrix = np.vstack([features_matrix, extract_features_from_tweet(tweet)])
    return features_matrix


def dump_features(features_matrix, filepath):
    np.savetxt(filepath, features_matrix, delimiter=";")


def extract_and_dump_features(input_filepath, output_filepath):
    feature_matrix = extract_features_from_tweets(open(input_filepath))
    dump_features(feature_matrix, output_filepath)


if __name__ == '__main__':
    extract_and_dump_features(TRAIN_TWEETS_FILE_NAME, TRAIN_FEATURES_FILE_NAME)
    extract_and_dump_features(TEST_TWEETS_FILE_NAME, TEST_FEATURES_FILE_NAME)
