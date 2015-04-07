__author__ = 'Yuliya'
from datetime import datetime
# FEATURE_EXTRACTORS
from samples import POS_TWEETS_FILE_NAME, NEG_TWEETS_FILE_NAME, DIR_TWEETS_FILES

smiles = [':)', ':-)', ': )', ':D', '=)', '))', ':(', ':-(', ': (', '=(', '((', ')', '(']

#удаляем лишние пробелы
def tweet_strip(line):
    line = line.strip().replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').replace('	', ' ')
    while line.find("  ") > -1:
        line = line.replace("  ", " ")
    return line


#удаляем смайлы, т.к. по ним проходила предварительная выборка
def delete_smile(line):
    for s in smiles:
        line = line.replace(s, '')
    return line


#убираем имя пользователя и ссылки на другие имена
def clear_of_name(line):
    #имя пользователя идет первым, поэтому просто удаляем всё до пробела
    s, t, line = line.partition(" ")
    #удаляем все ссылки
    while line.find('@') > -1:
        st, p, end = line.partition('@')
        s, p, end = end.partition(" ")
        line = ' '.join([st, end])
    return line


#убираем ссылки
def clear_of_link(line):
    #удаляем все ссылки
    while line.find('http') > -1:
        st, p, end = line.partition('http')
        s, p, end = end.partition(" ")
        line = ' '.join([st, ' *link* ', end])
    return line


#удаляем цифры
def delete_digits(line):
    trans_dict = {ord("{}".format(x)): "" for x in range(10)}
    return line.translate(trans_dict)


def clean_tweets(path, filename):
    txt_file = open(path + filename, encoding="utf8")
    txt_clean = open(''.join([path, 'clean_', filename]), 'w+', encoding="utf8")
    # st = datetime.now()
    # count = 0
    # th = 0
    for line in txt_file:
        line = tweet_strip(delete_digits(delete_smile(clear_of_link(clear_of_name(tweet_strip(line))))))
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
        line = tweet_strip(delete_digits(delete_smile(clear_of_link(clear_of_name(tweet_strip(line))))))
        if len(line):
            lines.add(line)
    for line in lines:
        txt_clean.write(line + '\n')


if __name__ == '__main__':
    clean_tweets_without_dubl(DIR_TWEETS_FILES, POS_TWEETS_FILE_NAME)
    clean_tweets_without_dubl(DIR_TWEETS_FILES, NEG_TWEETS_FILE_NAME)
