__author__ = 'Yuliya'

smiles = [':)', ':-)', ': )', ':D', '=)', '))', ':(', ':-(', ': (', '=(', '((', ')', '(']

#удаляем лишние пробелы
def tweet_strip(line):
    line = line.strip().replace('\r', ' ').replace('\t', ' ').replace('\n', ' ')
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
        line = ' '.join([st, end])
    return line


#удаляем цифры
def delete_digits(line):
    trans_dict = {ord("{}".format(x)): "" for x in range(10)}
    return line.translate(trans_dict)


def clean_tweets(path, filename):
    txt_file = open(path + filename, encoding="utf8")
    txt_clean = open(''.join([path, 'clean_', filename]), 'w+', encoding="utf8")
    for line in txt_file:
        line = tweet_strip(delete_digits(delete_smile(clear_of_link(clear_of_name(tweet_strip(line))))))
        if len(line):
            txt_clean.write(line + '\n')


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
    clean_tweets_without_dubl('../resources/', 'plus_smile.txt')
