# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: write_to_tfrecords.py
@time: 2018/7/3 14:38
"""

import argparse
import numpy as np
import os
import tensorflow as tf
import class_reader_loader as myloader

CLASS_DICT_FILE_PATH = 'preprocessing/classDict.txt'


def file_contents(FILE_PATH):
    contents = []
    with open(FILE_PATH, 'r', encoding='ascii') as f:
        for line in f:
            if line.strip() is not '':
                contents.append(line)
    return contents


def get_scat_file_name(file_content):
    music_path = file_content.split('\t')[0].strip()
    cur_scat_file_name = music_path.replace('/', '-') + '.scat'
    return cur_scat_file_name


def scattering_to_tfrecords(scattering_path, tfrecords_file, class_dict, flag):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    training_num = 0
    test_num = 0

    if flag=='training':

        all_scat_files = os.listdir(scattering_path)
        for f_ct in training_file_cont:
            scat_file_name = get_scat_file_name(f_ct)
            if scat_file_name in all_scat_files:
                cur_path = os.path.join(scattering_path, scat_file_name)

                if 'training' in tfrecords_file:
                    # get music class
                    music_name = scat_file_name[scat_file_name.rfind('-') + 1:].replace('.scat', '')
                    for ctent in training_file_cont:
                        if music_name in ctent:
                            music_class_name = ctent.split('\t')[1].strip()
                            music_class_num = np.int64(class_dict[music_class_name])

                    feature_bytes = np.loadtxt(cur_path, dtype=np.float32, delimiter=",").tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        # len=1
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[music_class_num])),
                        # (433, 157)
                        "features_scattering": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes])),

                    }))

                    writer.write(example.SerializeToString())
                    print("Successfully written into tfrecords: {}".format(
                        scat_file_name + '_' + music_class_name + '_' + str(music_class_num)))
                    training_num += 1
            else:
                raise FileExistsError('could not find file: {} in {}, please check results of feature extraction.'.format(scat_file_name, scattering_path))


    elif flag=='test':

        all_scat_files = os.listdir(scattering_path)
        for f_ct in test_file_cont:
            scat_file_name = get_scat_file_name(f_ct)
            if scat_file_name in all_scat_files:
                cur_path = os.path.join(scattering_path, scat_file_name)

                if 'test' in tfrecords_file:
                    feature_bytes = np.loadtxt(cur_path, dtype=np.float32, delimiter=",").tobytes()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        "features_scattering": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[feature_bytes])),

                    }))

                    writer.write(example.SerializeToString())
                    print("Successfully written into tfrecords: {}".format(scat_file_name))
                    test_num+=1

            else:
                raise FileExistsError(
                    'could not find file: {} in {}, please check results of feature extraction.'.format(scat_file_name, scattering_path))

    writer.close()
    print("###############  {} Finished.   #############".format(tfrecords_file))
    if 'training' in tfrecords_file:
        print('training_num %i' % training_num)
    if 'test' in tfrecords_file:
        print('test_num %i' % test_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write scattering coefficients to tfrecords file.')
    parser.add_argument('--pathOfScratchFolder', type=str, default='debug_data/',
                        help='path to scratch folder')
    parser.add_argument('--pathOfTrainFileList', type=str, default='debug_data/trainListFile.txt',
                        help='path to trainFileList.txt')
    parser.add_argument('--pathOfTestFileList', type=str, default='debug_data/testListFile.txt',
                        help='path to testFileList.txt')
    args = parser.parse_args()

    training_file_cont = file_contents(args.pathOfTrainFileList)
    test_file_cont = file_contents(args.pathOfTestFileList)
    class_dict = myloader.get_class_dict(CLASS_DICT_FILE_PATH)

    scat_coeff_path = os.path.join(args.pathOfScratchFolder, 'preprocessing','scat_coefficients')

    if not os.path.exists(os.path.join(args.pathOfScratchFolder,'preprocessing','data_tfrecords')):
        os.mkdir(os.path.join(args.pathOfScratchFolder,'preprocessing','data_tfrecords'))
    training_file = args.pathOfScratchFolder + '/preprocessing/data_tfrecords/scattering_training.tfrecords'
    test_file = args.pathOfScratchFolder + '/preprocessing/data_tfrecords/scattering_test.tfrecords'

    scattering_to_tfrecords(scat_coeff_path, training_file, class_dict, 'training')
    scattering_to_tfrecords(scat_coeff_path, test_file, class_dict, 'test')
