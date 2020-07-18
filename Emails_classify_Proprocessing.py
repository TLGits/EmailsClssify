import re
import os
import argparse
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def del_punctuation(content):
    """
    删除标点，保证单词之间仅有一个空格
    :param content: 文件内容
    :return: 文件内容
    """
    content = re.sub("[\\\'\t\n\.\!\/_,$%^*()\[\]+\"<>\-:]+|[|+——！，。？?、~@#￥%……&*（）]+", " ", content)
    content = ' '.join(content.split())
    return content

def stemming(content):
    """
    词干提取stemming
    :param content: 文件内容
    :return: 文件内容
    """
    stemmer = SnowballStemmer("english")  # 选择目标语言为英语
    all_words = content.split(' ')
    new_content = []
    for word in all_words:
        new_word = stemmer.stem(word.lower())  # Stem a word 并且转化为小写
        if new_word != ' ':
            new_content.append(new_word)
    return " ".join(new_content)

def get_wordnet_pos(tag):
    """
    获取单词的词性
    :param tag: 词性
    :return: 词类型
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatization(content):
    """
    词形还原 lemmatization
    :param content: 文件内容
    :return: 文件内容
    """
    all_words = word_tokenize(content)  # 分词
    tagged_sent = pos_tag(all_words)  # 获取单词词性

    wnl = WordNetLemmatizer()
    new_content = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        new_content.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
    return " ".join(new_content)

def remove_stop_words(content):
    """
    去停用词
    :param content: 文件内容
    :return: 文件内容
    """
    stopwords_list = stopwords.words('english')
    all_words = content.split(' ')
    new_content = []
    for word in all_words: 
        if word not in stopwords_list:
            new_content.append(word)
    return " ".join(new_content)

def tfidf(file_list, label):
    """
    特征提取 采用tfidf
    :param file_list: 合成的总文件
    :param label: 每个文件的类别标签
    :return: dataframe 表格
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(file_list)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df['@label'] = label

    print(df)
    return df

if __name__ == '__main__':

    # 可配置式方法选择
    parser = argparse.ArgumentParser(description='Choose the method(stemming or lemmatization)')
    parser.add_argument('--method', '-m', help='method 词干提取stemming或词性还原lemmatization，非必要参数，但有默认值', default = 'stemming')
    args = parser.parse_args()
    method = args.method

    file_r_list = []  # 存储所有的文件内容
    label = []  # 存储类别结果，'baseball':1; 'hockey':-1

    for type_name in ['baseball', 'hockey']:
        url = 'dataset/Case1-classification/' + type_name
        for file_name in os.listdir(url):
            try:
                file = open(url + '/' + file_name, 'r', encoding='latin-1')
                lines = file.readlines()
            except Exception as e:
                print(url + '/' + file_name + '无法打开')
                print(e)
                continue

            # 读取每一封邮件“第四行——末尾”，并存入content中
            content = ''
            for i in range(3, len(lines)):
                content += lines[i]

            # 数据预处理：1. 去除标点符号；2. 词性还原/词干提取; 3. 去除停用词
            content = del_punctuation(content)
            if method == 'stemming':
                content = stemming(content)
            elif method == 'lemmatization':
                content = lemmatization(content)
            content = remove_stop_words(content)

            file_r_list.append(content)

            # 数据打标签
            if type_name == 'baseball':
                label.append(1)
            elif type_name == 'hockey':
                label.append(-1)

    preed_data = tfidf(file_r_list, label)

    preed_data.to_csv(method + '_preed_data.csv', index = False)

