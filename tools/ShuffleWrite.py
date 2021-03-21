# coding:utf-8
# Copyright 2016 Google Inc. All Rights Reserved.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.
"""

from datetime import datetime
import os
import random
import sys
import threading
import shutil
import struct
from sklearn.model_selection import train_test_split
import numpy as np
# import tensorflow as tf
import logging
import glob
logging.basicConfig(level=logging.INFO)


def check_and_set_default_args(command_args):
    if not (hasattr(command_args, 'train_shards')) or command_args.train_shards is None:
        command_args.train_shards = 5
    # if not (hasattr(command_args, 'validation_shards')) or command_args.validation_shards is None:
    #     command_args.validation_shards = 5
    if not (hasattr(command_args, 'num_threads')) or command_args.num_threads is None:
        command_args.num_threads = 5
    if not (hasattr(command_args, 'class_label_base')) or command_args.class_label_base is None:
        command_args.class_label_base = 0
    if not (hasattr(command_args, 'dataset_name')) or command_args.dataset_name is None:
        command_args.dataset_name = ''
    if not (hasattr(command_args, 'slice_length')) or command_args.slice_length is None:
        command_args.dataset_name = 262144  # 2^16 个切割点数
    assert not command_args.train_shards % command_args.num_threads, (
        'Please make the command_args.num_threads commensurate with command_args.train_shards')
    assert not command_args.validation_shards % command_args.num_threads, (
        'Please make the command_args.num_threads commensurate with '
        'command_args.validation_shards')
    assert command_args.train_directory is not None
    assert command_args.validation_directory is not None
    assert command_args.labels_file is not None
    assert command_args.output_directory is not None
    # assert command_args.gaoxiao is not None


def _is_dat(filename):
    # 判断是否为dat文件
    return '.dat' in filename


def _process_dataset(name, directory, num_shards, labels_file, command_args):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: 训练集/ 验证集 名称
      directory: 路径
      num_shards: 线程数
      labels_file: 前面定义的label文件名
    """
    filenames, texts, labels = _find_image_files(directory, labels_file, command_args)
    # _process_image_files(name, filenames, texts, labels, num_shards, command_args)
    print(filenames, labels, texts)


def _find_image_files(data_dir, labels_file, command_args):
    """
    记录数据文件名、 对应的文件路径、 文件标签  然后给随机打散
    """
    logging.info('Determining list of input files and labels from %s.' % data_dir)
    # unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
    unique_labels = [line.strip() for line in open(labels_file, 'r', encoding='utf-8')]  # label=[work, sleep, phone]
    print('label is : ', unique_labels)
    # strip() 貌似用来去 回车的\n
    labels = []
    filenames = []
    texts = []
    label2index = {'working': 0, 'sleeping': 1, 'looking_at_phone': 2}

    # Leave label index 0 empty as a background class.
    """非常重要，这里我们调整label从0开始以符合定义"""
    # label_index = command_args.class_label_base

    # Construct the list of JPEG files and labels.
    # 这里进行 文件切分
    for text in unique_labels:  # 这里按目标类别循环
        jpeg_file_path = '%s/%s/*.mp4' % (data_dir, text)  # 这里限定了
        print('the mp4 file path is: ', jpeg_file_path)
        # jpeg_file_path_new = '%s/%s/%s/' % (data_dir, text, 'slice')
        # if os.path.exists(jpeg_file_path_new):  # 清空之前记录
        #     shutil.rmtree(jpeg_file_path_new)
        # os.mkdir(jpeg_file_path_new)
        matching_files = glob.glob(jpeg_file_path)
        count = len(matching_files)
        labels.extend([label2index[text]] * count)  # labels =[0,0,0,...0]
        texts.extend([text] * count)  # texts = [work, work, work,....]
        filenames.extend(matching_files)
        # label_index += 1

    # 数据随机化
    # saved TFRecord files. Make the randomization repeatable. 存储 TF文件   repeatable？
    shuffled_index = list(range(len(filenames)))  # 这里需要加上list ， range生成的数据不再是list格式了
    random.seed(12245)
    random.shuffle(shuffled_index)  # 对所有的路径进行shuffle打乱

    # 打乱顺序shuffle
    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    # 制作tsv文件，用于最后降维显示
    print("Making " + data_dir + " metadata……\n")
    path_for_metadata = os.path.join(command_args.data_dir, command_args.meta_file)
    with open(path_for_metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for i in range(len(shuffled_index)):
            # f.write("%d\t%d\n" % (shuffled_index[i], labels[i]))
            f.write("%d\t%d\n" % (i, labels[i]))
    print("Completed Making " + data_dir + " metadata!\n")

    logging.info('Found %d MP4 files across %d labels inside %s.Now make csv file.\n' %
                 (len(filenames), len(unique_labels), data_dir))

    train_csv_path = os.path.join(command_args.data_dir, command_args.train_csv)
    val_csv_path = os.path.join(command_args.data_dir, command_args.val_csv)
    test_csv_path = os.path.join(command_args.data_dir, command_args.test_csv)
    trainpath, valpath, trainlabel, vallabel = train_test_split(filenames, labels, test_size=0.15, random_state=0, shuffle=True)
    trainpath, testpath, trainlabel, testlabel = train_test_split(trainpath, trainlabel, test_size=0.2, random_state=0, shuffle=True)

    with open(train_csv_path, 'w') as f:
        for filename, label in zip(trainpath, trainlabel):
            content = "%s %d" % (filename, label)
            f.write(content + '\n')

    with open(val_csv_path, 'w') as f:
        for filename, label in zip(valpath, vallabel):
            content = "%s %d" % (filename, label)
            f.write(content + '\n')

    with open(test_csv_path, 'w') as f:
        for filename, label in zip(valpath, vallabel):
            content = "%s %d" % (filename, label)
            f.write(content + '\n')

    print(test_csv_path)

    return filenames, texts, labels


def _process_signal(filename):
    # Read the image file.
    with open(filename, 'rb') as f:
        signal_buffer = f.read()
        length_file = os.path.getsize(filename)
        assert not length_file % 4  # 必须整除
        length_str = ('=' + 'h' * int(length_file / 2))
        # print('hello')
        # signal_data = np.array([struct.unpack(length_str, signal_buffer)])
        signal_data = list(struct.unpack(length_str, signal_buffer))

    return signal_data


def main(command_args):
    """
    command_args:需要有以下属性：
    command_args.train_directory  训练集所在的文件夹。这个文件夹下面，每个文件夹的名字代表ch1 ch2名称，再下面就是数据。
    command_args.validation_directory 验证集所在的文件夹。这个文件夹下面，每个文件夹的名字代表ch1 ch2名称，再下面就是数据。
    command_args.labels_file 一个文件。每一行代表一个label名称。
    command_args.output_directory 一个文件夹，表示最后输出的位置。

    command_args.train_shards 将训练集分成多少份。
    command_args.validation_shards 将验证集分成多少份。
    command_args.num_threads 线程数。必须是上面两个参数的约数。

    command_args.class_label_base 很重要！真正的tfrecord中，每个class的label号从多少开始，默认为0（在models/slim中就是从0开始的）
    command_args.dataset_name 字符串，输出的时候的前缀。

    数据不可以有损坏。否则会导致线程提前退出。
    """
    check_and_set_default_args(command_args)  # 检查attribute 利用函数hasattr
    logging.info('Saving results to %s' % command_args.output_directory)

    # _process_dataset('validation', command_args.validation_directory,
    #                  command_args.validation_shards, command_args.labels_file, command_args)
    _process_dataset('train',
                     command_args.train_directory,
                     command_args.validation_shards,
                     command_args.labels_file,   # label.txt
                     command_args)
