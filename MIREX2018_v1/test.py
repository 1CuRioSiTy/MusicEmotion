# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: random_initial_learning.py
@time: 2018/7/4 15:01
"""

from __future__ import print_function, division

import argparse
import tensorflow as tf
import numpy as np
import os
import class_reader_loader as my_loader

from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from class_reader_loader import get_class_dict

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

x_height = 157
x_width = 433

CLASS_DICT_FILE_PATH = 'preprocessing/classDict.txt'
FILENUMBER_DICT_FILE_PATH = 'preprocessing/filenumberDict.txt'
class_dict = my_loader.get_class_dict(CLASS_DICT_FILE_PATH)
n_classes = len(class_dict) # [0, 2), e.g. 0,1  otherwise nan loss vals

batch_size = 1

filenum_dict = my_loader.get_meta_dict(FILENUMBER_DICT_FILE_PATH)
training_files = filenum_dict['training_num']
test_files = filenum_dict['test_num']

# 1 epoch needed iterations
training_iter = int(training_files/batch_size)
test_iter = int(test_files/batch_size)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
                                        1e-5,                # Base learning rate.
                                        global_step,         # Current index into the dataset.
                                        10000,          # Decay step.
                                        0.8,                # Decay rate.
                                        staircase=True)
training_iterations = 200 * training_iter # 200 epochs   900/20 * 200 = 9000
display_step = 1
state_size = 512
num_layers = 5
dropout = 1

# truncated_backprop_length
truncated_backprop_length = 433
# truncated_num
truncated_num = 157


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'features_scattering': tf.FixedLenFeature([], tf.string),
                                       })

    x = tf.decode_raw(features['features_scattering'], tf.float32)
    x = tf.reshape(x, [x_width, x_height])
    x = tf.transpose(x)
    x = tf.reshape(x, [x_height, x_width])
    return x

def load_batch_data(path, batch_size=batch_size):
    features = read_and_decode(path)
    audio_batch = tf.train.batch([features], batch_size=batch_size, allow_smaller_final_batch=True)
    return audio_batch


def is_zeros(arr):
    for element in arr:
        if element != 0:
            return False
    return True

def arr_2dims_to_1dim(arr):
    return [v[0] for v in arr]

def get_accuracy(tags, batch_logits):
    count = 0
    max_logit_index = np.argmax(batch_logits, axis=1)
    # print(max_logit_index)

    # print(tags)
    # print(max_logit_index)
    for i in range(batch_size):
        if tags[i][0] == max_logit_index[i]:
            count += 1
    return float(count) / batch_size

def get_batch_predictions(batch_logits):
    max_logit_index = np.argmax(batch_logits, axis=1)
    return max_logit_index

def get_rnn_variables_to_restore():
  return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'lstm_cell' in v.name]


def get_genre_name(class_num):
    class_dict = get_class_dict(CLASS_DICT_FILE_PATH)
    for k, v in class_dict.items():
        if v == class_num:
            return k


def write_to_outputListFile(test_file_path, output_path, predictions, print_result=True):
    cnt_of_writter = []

    with open(test_file_path, 'r', encoding='ascii') as f:
        counter = 0
        for line in f:
            if line.strip() is not '':
                cnt_of_writter.append(line.strip() + '\t' + get_genre_name(predictions[counter]) + '\n')
                counter += 1
        assert counter == len(predictions), 'the length of predictions should be equal with the line number of test file.'

    with open(output_path, 'w', encoding='ascii') as f:
        for cnt in cnt_of_writter:
            f.write(cnt)
            if print_result:
                print(cnt.strip())
        print('Predictions have been saved in file: {}'.format(output_path))

# placeholders
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_num, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int64, [batch_size])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([state_size, n_classes]), dtype=tf.float32),

}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32),
}

def RNN(X, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    X = tf.transpose(batchX_placeholder, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    X = tf.reshape(X, [-1, truncated_backprop_length])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    X = tf.split(X, truncated_num, 0)

    cell = MultiRNNCell([DropoutWrapper(LSTMCell(state_size), output_keep_prob=dropout) for _ in range(num_layers)])

    # Forward passes
    outputs, current_state = tf.contrib.rnn.static_rnn(cell, X, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'] #, outputs, current_state
#RNN_results = RNN(batchX_placeholder, weights, biases, rnn_tuple_state_initial)
#logits = RNN_results[0]
#outputs = RNN_results[1]
#current_state = RNN_results[2]
logits = RNN(batchX_placeholder, weights, biases)

cross_entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batchY_placeholder)
mean_batch_loss = tf.reduce_mean(cross_entropy_losses)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_batch_loss, global_step=global_step)

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--pathOfScratchFolder', type=str, default='debug_data/',
                    help='path to scratch folder')
parser.add_argument('--pathOfTestFileList', type=str, default='debug_data/testListFile.txt',
                    help='path to testFileList.txt')
parser.add_argument('--pathOfOutput', type=str, default='debug_data/outputListFile.txt',
                    help='path to outputListFile.txt')
args = parser.parse_args()

test_tfrecord_file_path = os.path.join(args.pathOfScratchFolder, 'preprocessing','data_tfrecords','scattering_test.tfrecords')
audio_batch_test = load_batch_data(test_tfrecord_file_path, batch_size)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    model_path = os.path.join(args.pathOfScratchFolder,'preprocessing','model','model_512_5l_LSTM.ckpt')
    saver.restore(sess, model_path)
    print('model has been loaded from {}'.format(model_path))
    # Test model
    # batch_test --> reduce_mean --> final_test_accuracy

    test_iterations = test_iter
    # test_accuracy_final = 0.

    all_predictions = []
    for _ in range(test_iterations):
        audio_test_vals = sess.run(audio_batch_test)
        logits_test = sess.run(logits, feed_dict={batchX_placeholder: audio_test_vals})
        _batch_predictions = get_batch_predictions(logits_test)
        all_predictions.extend(_batch_predictions.tolist())
        print("test iter: %d, test batch predictions: %s" % (_, _batch_predictions))
        # test_accuracy = get_accuracy(label_test_vals, logits_test)
        # test_accuracy_final += test_accuracy
        # print("test epoch: %d, test loss: %f, test accuracy: %f" % (_, test_loss_val, test_accuracy))
    # test_accuracy_final /= test_iterations
    
    # print("final test accuracy: %f" % test_accuracy_final)
    print("all test predictions: %s" % all_predictions)

    write_to_outputListFile(args.pathOfTestFileList, args.pathOfOutput, all_predictions)
    coord.request_stop()
    coord.join(threads)
    sess.close()
