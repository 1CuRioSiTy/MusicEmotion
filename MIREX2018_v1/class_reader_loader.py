# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: class_reader_loader.py
@time: 2018/7/5 13:28
"""

import argparse

def write_class_index_to_file(training_file_path, class_dict_file_path):
    class_dict = {}
    n = 0 # class index


    with open(training_file_path, 'r', encoding='ascii') as f:

        for line in f:
            if line.strip() is not '':

                class_name = line[line.index('\t'):].strip()
                if not class_name in class_dict:
                    class_dict[class_name] = n
                    n += 1

    print(class_dict)

    with open(class_dict_file_path, 'w', encoding='ascii') as f:
        f.write(str(class_dict))


def get_class_dict(class_dict_file_path):
    with open(class_dict_file_path, 'r', encoding='ascii') as f:
        a = f.read()
        assert a is not '', 'Class dict file is empty. excute the write function first.'
        return eval(a)

def file_counter(file_path):
    counter = 0
    with open(file_path, 'r', encoding='ascii') as f:

        for line in f:
            if line.strip() is not '':

                counter += 1
    return counter

def save_training_test_file_meta(filenumber_dict_path,training_filelist,test_filelist):
    filenumber_dict={}
    filenumber_dict['training_num']=file_counter(training_filelist)
    filenumber_dict['test_num']=file_counter(test_filelist)

    with open(filenumber_dict_path, 'w', encoding='ascii') as f:
        f.write(str(filenumber_dict))

def get_meta_dict(filenumber_dict_path):
    with open(filenumber_dict_path, 'r', encoding='ascii') as f:
        dict = f.read()
        return eval(dict)

if __name__ == '__main__':
    CLASS_DICT_FILE_PATH = 'preprocessing/classDict.txt'
    FILENUMBER_DICT_FILE_PATH = 'preprocessing/filenumberDict.txt'

    parser = argparse.ArgumentParser(description='Write meta data to files.')
    parser.add_argument('--pathOfTrainFileList', type=str, default='debug_data/trainListFile.txt',
                        help='path to trainFileList.txt')
    parser.add_argument('--pathOfTestFileList', type=str, default='debug_data/testListFile.txt',
                        help='path to testFileList.txt')
    args = parser.parse_args()

    write_class_index_to_file(args.pathOfTrainFileList, CLASS_DICT_FILE_PATH)
    # print(get_class_dict(CLASS_DICT_FILE_PATH))
    save_training_test_file_meta(FILENUMBER_DICT_FILE_PATH, args.pathOfTrainFileList, args.pathOfTestFileList)
    print(get_meta_dict(FILENUMBER_DICT_FILE_PATH))
