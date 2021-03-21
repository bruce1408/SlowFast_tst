# -*- coding: utf-8 -*-
import argparse
import os
from gensim.corpora import WikiCorpus
import jieba
# import tensorflow as tf
import glob
from string import punctuation


# from langconv import *

def self_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--csv-path', default='E:/Project/Pro_KineticsCSV/Kinetics/data/kinetics-600_train.csv')
    parser.add_argument('--meta-file', default='Sig_Meta.tsv', type=str)
    parser.add_argument('--train-shards', default=5, type=int)
    parser.add_argument('--validation-shards', default=5, type=int)
    parser.add_argument('--num-threads', default=5, type=int)  # 分类数shard 要是线程的倍数
    parser.add_argument('--dataset-name', default='Hello', type=str)
    parser.add_argument('--slice-length', default=4096,
                        type=int)  # 65536=128*128*2*2 每个文件大小 32*32*2*2=4096  64*64*2*2=16384
    parser.add_argument('--start-offset', default=10000, type=int)
    parser.add_argument('--max-num-sample', default=2048, type=int)  # 同一个文件最大截取个数1024 4096对应32窗
    return parser.parse_args()


def get_content(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l
        return content


def my_function():
    args = self_args()
    csv = args.csv_path
    i = 0
    str = ""
    for line in open(csv, 'r', encoding='utf-8'):
        if i == 0:
            str = str + line
            i = i + 1
            continue;
        tmp = line.split(',')[0]
        if (tmp == "sleeping" or tmp == "looking at phone" or tmp == "talking on cell phone"):
            str = str + line
        i = i + 1

    csv_new = csv + "_"
    with open(csv_new, 'w', encoding='utf-8') as f:
        f.write(str + "\n")
    #         matching_files = tf.gfile.Glob(file_path)
    #         for matching_filename in matching_files:
    #             filename_ = matching_filename.split('\\')[-1]
    #             with open(matching_filename, 'r') as f:
    #
    #
    # space = ' '
    # i = 0
    # l = []
    # zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    # f = open('./data/reduce_zhiwiki.txt', 'w')
    # wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    # for text in wiki.get_texts():
    #     for temp_sentence in text:
    #         temp_sentence = Converter('zh-hans').convert(temp_sentence)
    #         seg_list = list(jieba.cut(temp_sentence))
    #         for temp_term in seg_list:
    #             l.append(temp_term)
    #     f.write(space.join(l) + '\n')
    #     l = []
    #     i = i + 1
    #
    #     if (i %200 == 0):
    #         print('Saved ' + str(i) + ' articles')
    # f.close()


if __name__ == '__main__':
    my_function()
