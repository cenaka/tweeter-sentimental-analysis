from samples import TRAIN_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME

POSITIVE_CLASS = "pos"
NEGATIVE_CLASS = "neg"
__author__ = 'cenaka'

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import datetime
def fit_svc(feature_matrix, answers):
    svc = svm.SVC(gamma=0.001, C=100., verbose=True)
    return svc.fit(feature_matrix, answers)


def read_feature_matrix_from_file(file_path):
    return np.loadtxt(file_path)


def generate_traning_answers():
    """
    Обучающая и тестовая выборки из примера содержат по 100 твитов: первые 50 негативных, вторые 50 позитивных
    """
    return np.concatenate((np.array([POSITIVE_CLASS] * 100000), np.array([NEGATIVE_CLASS] * 100000)))

def generate_test_answers():
    """
    Обучающая и тестовая выборки из примера содержат по 100 твитов: первые 50 негативных, вторые 50 позитивных
    """
    return np.concatenate((np.array([POSITIVE_CLASS] * 100000), np.array([NEGATIVE_CLASS] * 100000)))

def evaluate(classifier, test_features, answers):
    predicted = [classifier.predict(test)[0] for test in test_features]
    target_names = [POSITIVE_CLASS, NEGATIVE_CLASS]
    print(classification_report(answers, predicted, target_names=target_names))


if __name__ == "__main__":
    time = datetime.datetime.now()
    classifier = fit_svc(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_traning_answers())
    print("fit model: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
    test_features = read_feature_matrix_from_file(TEST_FEATURES_FILE_NAME)
    answers = generate_test_answers()
    evaluate(classifier, test_features, answers)