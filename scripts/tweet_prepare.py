__author__ = 'Yuliya'
import pymorphy2
import re

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


#удаляем цифры
def delete_digits(line):
    trans_dict = {ord("{}".format(x)): "" for x in range(10)}
    return line.translate(trans_dict)


#удаляем повторения букв "аааа"
def delete_repeat(line):
    while bool(re.compile(r"([a-zA-Zа-яА-Я])\1").search(line)):
        line = re.sub(r"([a-zA-Zа-яА-Я])\1", r"\1", line)
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
    return normalized_tweet(delete_repeat(tweet_strip(delete_non_letter(clear_of_link(clear_of_name(tweet_strip(tweet)))))))


def clean_tweets(path, filename):
    txt_file = open(path + filename, encoding="utf8")
    txt_clean = open(''.join([path, 'clean_', filename]), 'w+', encoding="utf8")
    # st = datetime.now()
    # count = 0
    # th = 0
    for line in txt_file:
        line = tweet_strip(delete_repeat(delete_non_letter(clear_of_link(clear_of_name(tweet_strip(line))))))
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
        line = tweet_strip(delete_repeat(delete_digits(delete_smile(clear_of_link(clear_of_name(tweet_strip(line)))))))
        if len(line):
            lines.add(line)
    for line in lines:
        txt_clean.write(line + '\n')


if __name__ == '__main__':
    #clean_tweets_without_dubl(DIR_TWEETS_FILES, POS_TWEETS_FILE_NAME)
    #clean_tweets_without_dubl(DIR_TWEETS_FILES, NEG_TWEETS_FILE_NAME)
    morph = pymorphy2.MorphAnalyzer()

    a = clean_tweet("savva601	@KSyomin  ага, а перед разобранным мостом ещё лежит, для ускорения, огромная куча дерьма в виде... а под мостом кол в виде БРИКС и ШОС! :-))")
    print(a)
