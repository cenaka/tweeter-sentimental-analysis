__author__ = 'Yuliya'
import pymorphy2
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from scipy import sparse, io
import datetime

from datetime import datetime
# FEATURE_EXTRACTORS

from samples import POS_TWEETS_FILE_NAME, NEG_TWEETS_FILE_NAME, DIR_TWEETS_FILES
smiles = [':)', ':-)', ': )', ':D', '=)', '))', ':(', ':-(', ': (', '=(', '((', ')', '(']
morph = pymorphy2.MorphAnalyzer()
DICTIONARY = {}

# удаляем лишние пробелы
def tweet_strip(line):
    return ' '.join(line.split())

# удаляем смайлы, т.к. по ним проходила предварительная выборка
def delete_smile(line):
    for s in smiles:
        line = line.replace(s, '')
    return line


# убираем имя пользователя и ссылки на другие имена
def clear_of_name(line):
    #имя пользователя идет первым, поэтому просто удаляем всё до пробела
    s, t, line = line.partition(" ")
    #удаляем все ссылки на другие имена
    return re.sub(r"@[^\s]+", " ", line)


#убираем ссылки
def clear_of_link(line):
    #удаляем все ссылки
    return re.sub(r"http[^\s]+", " ", line)


#склеиваем частицу "не" со словом, идущим после нее
def glue_of_no(line):
    return re.sub(r"\bне\s", "не", line)


#убираем предлоги, союзы и т.п.
def clear_of_particles(line):
    return re.sub(r"\bи\b|\bа\b|\bну\b|\bперед\b|\bпосле\b|\bчерез\b|\bза\b|\bнад\b|\bно\b|\bв\b|\bпод\b|\bу\b|\bк\b|\bс\b|\bпо\b|\bдо\b|\bбы\b|\bни\b|\bвот\b|\bна\b|\bво\b|\bли\b|\bсо\b", "", line)


#удаляем цифры
def delete_digits(line):
    trans_dict = {ord("{}".format(x)): "" for x in range(10)}
    return line.translate(trans_dict)


#удаляем повторения букв "аааа"
def delete_repeat(line):
    while bool(re.compile(r"([a-zA-Zа-яА-Я])\1\1").search(line)):
        line = re.sub(r"([a-zA-Zа-яА-Я])\1\1", r"\1\1", line)
    return line


# удаляем все кроме пробелов и букв (удаленное заменяем одинарными пробелами)
def delete_non_letter(line):
    return re.sub(r"[^\s\w]+|\d+", " ", line)


def normalized_word(word):
    if DICTIONARY.get(word):
        return DICTIONARY.get(word)
    else:
        normal_form = morph.parse(word)[0].normal_form
        DICTIONARY[word] = normal_form
        return normal_form


def normalized_tweet(tweet):
    return " ".join(map(lambda w: normalized_word(w), tweet.split()))


def clean_tweet(tweet):
    return tweet_strip(clear_of_particles(glue_of_no(normalized_tweet(delete_repeat(delete_non_letter(clear_of_link(clear_of_name(tweet_strip(tweet)))))))))


def clean_tweets(path, filename):
    txt_file = open(path + filename, encoding="utf8")
    txt_clean = open(''.join([path, 'clean_', filename]), 'w+', encoding="utf8")
    # st = datetime.now()
    # count = 0
    # th = 0
    for line in txt_file:
        line = tweet_strip(clear_of_particles(glue_of_no(delete_repeat(delete_non_letter(clear_of_link(clear_of_name(tweet_strip(line))))))))
        if len(line):
            txt_clean.write(line + '\n')
        # count += 1
        # if count > 10000:
        #     th += 1
        #     count = 0
        #     print(''.join(["count", str(th), "0 - ", str((datetime.now() - st).seconds)]))


def clean_tweets_without_dubl(path, filename):
    txt_file = open(path + filename, encoding="utf8")
    txt_clean = open(''.join([path, 'clean_', filename]), 'w+', encoding="utf8")
    lines = set()
    for line in txt_file:
        line = tweet_strip(clear_of_particles(glue_of_no(delete_repeat(delete_digits(delete_smile(clear_of_link(clear_of_name(tweet_strip(line)))))))))
        if len(line):
            lines.add(line)
    for line in lines:
        txt_clean.write(line + '\n')


if __name__ == '__main__':
    #clean_tweets_without_dubl(DIR_TWEETS_FILES, POS_TWEETS_FILE_NAME)
    #clean_tweets_without_dubl(DIR_TWEETS_FILES, NEG_TWEETS_FILE_NAME)
    morph = pymorphy2.MorphAnalyzer()

    a = clean_tweet("AndrewZu	@blyelldill боюсь спросить, почему они зеленые?))")
    print(a)
    vectorSK = CountVectorizer(min_df=1)
    features_matrix = vectorSK.fit([a])
    print(features_matrix.get_feature_names())

    vectorSK = CountVectorizer(ngram_range=(2, 2), min_df=1)
    features_matrix = vectorSK.fit([a])
    print(features_matrix.get_feature_names())

    vectorSK = CountVectorizer(ngram_range=(4, 4), analyzer="char", min_df=1)
    features_matrix = vectorSK.fit([a])
    print(features_matrix.get_feature_names())
