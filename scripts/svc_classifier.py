from samples import TRAIN_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME, CLASSIFIER, TRAIN_OTHER_FEATURES_FILE_NAME, TEST_OTHER_FEATURES_FILE_NAME

POSITIVE_CLASS = "pos"
NEGATIVE_CLASS = "neg"
__author__ = 'cenaka'

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import datetime
from sklearn.externals import joblib
from scipy import io
N = 100000

def fit_svc(feature_matrix, answers):
    svc = svm.SVC(gamma=0.001, C=100., verbose=True)
    return svc.fit(feature_matrix, answers)


def read_feature_matrix_from_file(file_path):
    return io.mmread(file_path)


def generate_training_answers():
    """
    Обучающая и тестовая выборки из примера содержат по 100 твитов: первые 50 негативных, вторые 50 позитивных
    """
    return np.concatenate((np.array([POSITIVE_CLASS] * N), np.array([NEGATIVE_CLASS] * N)))


def generate_test_answers():
    """
    Обучающая и тестовая выборки из примера содержат по 100 твитов: первые 50 негативных, вторые 50 позитивных
    """
    return np.concatenate((np.array([POSITIVE_CLASS] * N), np.array([NEGATIVE_CLASS] * N)))


def evaluate(classifier, test_features, answers):
    predicted = classifier.predict(test_features)
    target_names = [POSITIVE_CLASS, NEGATIVE_CLASS]
    print(classification_report(answers, predicted, target_names=target_names))


if __name__ == "__main__":
    time = datetime.datetime.now()
    classifier = fit_svc(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers())

    joblib.dump(classifier, CLASSIFIER)
    #classifier = joblib.load(CLASSIFIER)
    print("fit model: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
    time = datetime.datetime.now()
    test_features = read_feature_matrix_from_file(TEST_FEATURES_FILE_NAME)

    answers = generate_test_answers()
    evaluate(classifier, test_features, answers)
    print("evaluate answers: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())