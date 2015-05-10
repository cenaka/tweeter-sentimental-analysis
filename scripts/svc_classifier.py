from sklearn.svm import SVC
from samples import TRAIN_FEATURES_FILE_NAME, TEST_FEATURES_FILE_NAME, NB_CLASSIFIER, SVC_CLASSIFIER, \
    TRAIN_UNIGRAMMS_FEATURES_FILE_NAME, TEST_UNIGRAMMS_FEATURES_FILE_NAME

POSITIVE_CLASS = "pos"
NEGATIVE_CLASS = "neg"
__author__ = 'cenaka'

import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import datetime
from sklearn.externals import joblib
from scipy import io
N = 4000000
M = 100000

def fit_svc(feature_matrix, answers):
    svc = svm.SVC(gamma=0.001, C=100., verbose=True)
    return svc.fit(feature_matrix, answers)


def fit_nb(feature_matrix, answers):
    gnb = MultinomialNB()
    return gnb.fit(feature_matrix, answers)


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
    return np.concatenate((np.array([POSITIVE_CLASS] * M), np.array([NEGATIVE_CLASS] * M)))


def evaluate(classifier, test_features, answers):
    predicted = classifier.predict(test_features)
    target_names = [POSITIVE_CLASS, NEGATIVE_CLASS]
    print(classification_report(answers, predicted, target_names=target_names))


def parameter_estimation(training_features, training_answers):
    parameters = [{'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
    scores = ['recall']
    #svr = svm.SVC()
    #clf = grid_search.GridSearchCV(svr, parameters)
    #clf.fit(training_features, training_answers)
    for score in scores:
        clf = GridSearchCV(SVC(C=1), parameters, cv=5, scoring='%s' % score)
        clf.fit(training_features, training_answers)
        print(clf.best_params_)

if __name__ == "__main__":
    time = datetime.datetime.now()
    #parameter_estimation(TRAIN_FEATURES_FILE_NAME, generate_test_answers())
    #classifier = fit_svc(read_feature_matrix_from_file(TRAIN_FEATURES_FILE_NAME), generate_training_answers())
    classifier = fit_nb(read_feature_matrix_from_file(TRAIN_UNIGRAMMS_FEATURES_FILE_NAME), generate_training_answers())
    joblib.dump(classifier, NB_CLASSIFIER)
    #classifier = joblib.load(NB_CLASSIFIER)
    print("fit model: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())
    time = datetime.datetime.now()
    test_features = read_feature_matrix_from_file(TEST_UNIGRAMMS_FEATURES_FILE_NAME)
    answers = generate_test_answers()
    evaluate(classifier, test_features, answers)
    print("evaluate answers: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())