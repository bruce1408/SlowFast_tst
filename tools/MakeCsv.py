# coding:utf-8

# 尝试从文件夹中将数据转化成tf格式数据
# 文件格式：  train两个文件夹 ch1 和 ch2 ， validation相同； 为 FMT信号化后的数据
# 本程序主要是 文件路径的初始化 存于args中；  主程序入口 tfrecord main函数
import argparse
import os
import logging
import random
from ShuffleWrite import main
# from dir_slice import main


def self_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--data-dir', default='./data202004gov')
    # parser.add_argument('-t', '--data_dir', default='/home/chenxi/SlowFast_tst/renameData')
    # parser.add_argument('-t', '--data_dir', default='/home/chenxi/dataset/train_data/train_data')
    parser.add_argument("-t", "--data_dir", default="/home/bruce/bigVolumn/autolabelData/train_data/train_data")

    parser.add_argument('--meta-file', default='Sig_Meta.tsv', type=str)
    parser.add_argument('--train-csv', default='train.csv', type=str)
    parser.add_argument('--val-csv', default='val.csv', type=str)
    parser.add_argument("--test-csv", default='test.csv', type=str)
    parser.add_argument('--train-shards', default=5, type=int)
    parser.add_argument('--validation-shards', default=5, type=int)
    parser.add_argument('--num-threads', default=5, type=int)   # 分类数shard 要是线程的倍数
    parser.add_argument('--dataset-name', default='Hello', type=str)
    parser.add_argument('--slice-length', default=512, type=int)
    parser.add_argument('--start-offset', default=10000, type=int)
    parser.add_argument('--max-num-sample', default=300, type=int)  # 同一个文件最大截取个数1024 4096对应32窗
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = self_args()
    args.meta_file = args.meta_file
    # args.train_directory = os.path.join(args.data_dir, 'train')
    args.train_directory = args.data_dir
    args.validation_directory = os.path.join(args.data_dir, 'validation')
    args.output_directory = args.data_dir
    args.labels_file = os.path.join(args.data_dir, 'label.txt')
    all_entries = ['working', 'sleeping', 'looking_at_phone']
    #  功能就是将 训练集的文件夹名字写在 lable.txt 中
    if os.path.exists(args.labels_file) is False:
        logging.warning('Can\'t find label.txt. Now create it')
        all_entries = os.listdir(args.train_directory)
        dirnames = []  # 这里默认的是 list类型
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w') as f:  # 对train数据集里面的目录获得目录名的一个列表
            for dirname in dirnames:
                f.write(dirname + '\n')
    # random.seed(11110)  # seed 是什么意思
    # args.start_offset = args.start_offset + random.randint(0, args.slice_length/4)*4  # 这里得到数据应该是4的倍数

    main(args)
# 正确的回车 \n


