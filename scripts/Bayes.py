__author__ = 'Yuliya'

import marisa_trie
import numpy as np
import pyximport
import os
from itertools import chain
from Stemmer import Stemmer
from tabulate import tabulate


pyximport.install(setup_args={"include_dirs": np.get_include()})


def compress(str):
    prev = ""
    res = []
    for x in str:
        if x != prev:
            prev = x
            res.append(x)
    return res


class TrieVectorizer:
    def __init__(self):
        self._vocabulary = None
        self._stemmer = Stemmer("russian")

    def fit(self, docs):
        # строит словарь по набору документов
        self._vocabulary = marisa_trie.Trie(
            chain(*[self._process(d) for d in docs]))
        return self

    def transform(self, docs):
        # векторизует набор документов с использованием словаря
        res = []
        list_len = len(self._vocabulary)
        for doc in docs:
            tmp = [0] * (list_len + 1)
            for word in self._process(doc):
                if word in self._vocabulary:
                    tmp[self._vocabulary[word]] += 1
                else:
                    tmp[list_len] += 1
            res.append(tmp)
        return res

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def _process(self, doc):
        return self._tokenize(self._analyze(doc))

    def _analyze(self, doc):
        # возвращает последовательность слов из документа
        return map(lambda word: word.strip().lower(), doc.split())

    def _tokenize(self, words, min_len=4):
        # превращает слова в "токены" длины не менее ``min_len``.
        tokens = []
        for word in words:
            if not word.isalpha():
                word = "".join(filter(lambda ch: ch.isalpha(), words))
            word = self._stemmer.stemWord("".join(compress(word)))
            if len(word) >= min_len:
                tokens.append(word)
        return tokens


class NaiveBayes:
    # @profile
    def fit(self, X, y):
        n_docs, n_words = len(X), len(X[0])
        n_classes = 2
        npX = np.array(X).T
        epsilon = np.finfo(float).eps
        class_prob = np.full(n_classes, .5)
        feature_prob = np.zeros((n_classes, n_words))
        for c in range(n_classes):
            class_prob[c] = y.count(c)
            feature_prob[c] = np.sum(npX * [(y[d] == c) for d in range(n_docs)], axis=1) + epsilon
        self._class_prob = class_prob / class_prob.sum()
        self._feature_prob = feature_prob / np.sum(feature_prob, axis=0)
        return self

    # @profile
    def predict(self, X):
        n_classes = 2
        npX = np.array(X)
        n_docs = len(X)
        log_feature_prob = np.log(self._feature_prob)
        log_class_prob = np.log(self._class_prob)
        y = np.zeros((n_classes, n_docs))
        for c in range(n_classes):
            y[c] = np.sum(npX * log_feature_prob[c], axis=1) + log_class_prob[c]
        return np.argmax(y, axis=0)


def train_test_split(X, y, ratio=0.2):
    mask = np.random.binomial(1, ratio, len(X)) == 1
    return X[mask], X[~mask], y[mask], y[~mask]


def classification_report(y_true, y_pred, labels):
    def precision_recall_total(y_true, y_pred):
        tp = (y_pred * y_true).sum()
        fp = (y_pred * ~y_true).sum()
        fn = (~y_pred * y_true).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall, int(y_true.sum())

    acc = []
    for c, label in enumerate(labels):
        precision, recall, total = precision_recall_total(
            np.asarray(y_true) == c, np.asarray(y_pred) == c)
        acc.append([label, precision, recall, total])

    print(tabulate(acc, headers=["", "precision", "recall", "total"],
                   floatfmt=".2f"))


if __name__ == "__main__":
    # 1. Load data.
    dir = '../resources/'
    pos_docs = list(open(os.path.join(dir, "clean_plus_smile.txt"),encoding="utf-8").readlines())
    neg_docs = list(open(os.path.join(dir, "clean_minus_smile.txt"),encoding="utf-8").readlines())
    pos_docs_test = list(open(os.path.join(dir, "clean_test_plus_smile.txt"),encoding="utf-8").readlines())
    if len(pos_docs) < len(pos_docs_test):
        pos_docs_test = pos_docs_test[:len(pos_docs)]
    else:
        pos_docs = pos_docs[:len(pos_docs_test)]
    neg_docs_test = list(open(os.path.join(dir, "clean_test_minus_smile.txt"),encoding="utf-8").readlines())
    if len(neg_docs) < len(neg_docs_test):
        neg_docs_test = neg_docs_test[:len(neg_docs)]
    else:
        neg_docs = neg_docs[:len(neg_docs_test)]
    docs_train = np.array(pos_docs + neg_docs)
    y_train = np.zeros(len(docs_train), dtype=np.bool)
    y_train[len(pos_docs):] = True
    docs_test = np.array(pos_docs_test + neg_docs_test)
    y_test = np.zeros(len(docs_test), dtype=np.bool)
    y_test[len(pos_docs_test):] = True

    # 2. Preprocess data.
    v = TrieVectorizer()
    X_train = v.fit_transform(docs_train)
    X_test = v.transform(docs_test)

    # # 3. Fit model and analyze performance.
    clf = NaiveBayes().fit(X_train, y_train.tolist())
    y_pred = np.array(clf.predict(X_test))
    classification_report(y_test, y_pred, ["pos", "neg"])
