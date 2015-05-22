from sklearn.svm import SVC
from samples import TRAIN_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME, NB_CLASSIFIER, SVC_CLASSIFIER, \
    UNIGRAMMS_TRAIN_FEATURES_FILE_NAME, UNIGRAMMS_TEST_FEATURES_FILE_NAME, BIGRAMMS_TRAIN_FEATURES_FILE_NAME,\
    BIGRAMMS_TEST_FEATURES_FILE_NAME, TRAIN_OTHER_FEATURES_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME, PREDICTED_TEST,\
    PREDICTED_TRAINING

POSITIVE_CLASS = "pos"
NEGATIVE_CLASS = "neg"
NEUTRAL_CLASS = "neut"
__author__ = 'cenaka'

import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import datetime
from scipy import io, sparse
import scipy.sparse as sp


def fit_svc(feature_matrix, answers):
    svc = svm.SVC(gamma=0.001, C=100., verbose=True)
    return svc.fit(feature_matrix, answers)


def fit_nb(feature_matrix, answers):
    gnb = MultinomialNB()
    return gnb.fit(feature_matrix, answers)


def read_feature_matrix_from_file(file_path):
    return io.mmread(file_path)


def generate_training_answers(N):
    return np.concatenate((np.array([POSITIVE_CLASS] * N), np.array([NEGATIVE_CLASS] * N),
                           np.array([NEUTRAL_CLASS] * N)))


def generate_test_answers(M):
    return np.concatenate((np.array([POSITIVE_CLASS] * 235), np.array([NEGATIVE_CLASS] * 702),
                           np.array([NEUTRAL_CLASS] * 1061)))


def evaluate(classifier, test_features, answers):
    predicted = classifier.predict(test_features)
    #np.savetxt("resources/probs.txt", predicted)
    target_names = [NEGATIVE_CLASS, NEUTRAL_CLASS, POSITIVE_CLASS]
    print(classification_report(answers, predicted, target_names=target_names))


if __name__ == "__main__":
    time = datetime.datetime.now()
    #classifier = fit_svc(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers())

    classifier = fit_nb(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers(2000000))

    print("fit nb model: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
    time = datetime.datetime.now()
    test_features = read_feature_matrix_from_file(TEST_FEATURES_FILE_NAME)
    answers = generate_test_answers(100000)
    evaluate(classifier, test_features, answers)
    print("evaluate answers: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())