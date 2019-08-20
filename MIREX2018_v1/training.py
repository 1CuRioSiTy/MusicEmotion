# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: random_initial_learning.py
@time: 2018/7/3 15:28
"""

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import os
import argparse
import time
import class_reader_loader as myloader

from math import ceil
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
# cpu only
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

x_height = 157
x_width = 433

CLASS_DICT_FILE_PATH = 'preprocessing/classDict.txt'
FILENUMBER_DICT_FILE_PATH = 'preprocessing/filenumberDict.txt'
class_dict = myloader.get_class_dict(CLASS_DICT_FILE_PATH)
n_classes = len(class_dict) # [0, 2), e.g. 0,1  otherwise nan loss vals

batch_size = 16 # 16

filenum_dict = myloader.get_meta_dict(FILENUMBER_DICT_FILE_PATH)
training_files = filenum_dict['training_num']
# test_files = 4

# 1 epoch needed iterations
training_iter = int(ceil(training_files/batch_size))
# test_iter = int(test_files/batch_size)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
                                        1e-5,                # Base learning rate.
                                        global_step,         # Current index into the dataset.
                                        10000,          # Decay step.
                                        0.8,                # Decay rate.
                                        staircase=True)
training_iterations = 160 * training_iter # 160 epochs
display_step = 500 # 500 iter
state_size = 512
num_layers = 5
dropout = 0.75

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
                                           'label': tf.FixedLenFeature([1], tf.int64),
                                       })

    x = tf.decode_raw(features['features_scattering'], tf.float32)
    x = tf.reshape(x, [x_width, x_height])
    x = tf.transpose(x)
    x = tf.reshape(x, [x_height, x_width])
    y = tf.cast(features['label'], tf.int64)
    return x, y

def load_and_shuffle_to_batch_data(path, batch_size=batch_size):
    features, label = read_and_decode(path)
    # 使用shuffle_batch可以随机打乱输入
    audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                      batch_size=batch_size, capacity=2000,
                                                      min_after_dequeue=1000, allow_smaller_final_batch=True)
    return audio_batch, label_batch


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

    # print(tags)
    # print(max_logit_index)
    for i in range(batch_size):
        if tags[i][0] == max_logit_index[i]:
            count += 1
    return float(count) / batch_size

def get_batch_predictions(batch_logits):
    max_logit_index = np.argmax(batch_logits, axis=1)
    return max_logit_index

# placeholders
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_num, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int64, [batch_size])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unstack(init_state, axis=0)  # num_layers 个 2 * batch_size * state_size
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
    #  output_keep_prob和input_keep_prob的区别？

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


parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--pathOfScratchFolder', type=str, default='debug_data/',
                    help='path to scratch folder')
args = parser.parse_args()
train_tfrecord_file_path = os.path.join(args.pathOfScratchFolder, 'preprocessing','data_tfrecords','scattering_training.tfrecords')

audio_batch_training, label_batch_training = load_and_shuffle_to_batch_data(train_tfrecord_file_path, batch_size)
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    valdation_accuracy_final = 0.
    time_start = time.time()
    for iteration_idx in range(training_iterations):

        audio_batch_vals_training, label_batch_vals_training = sess.run([audio_batch_training, label_batch_training])
        # _, loss_val, pred_, _current_state, _outputs = sess.run([train_step, mean_batch_loss, logits, current_state, outputs],
        _, loss_val, pred_ = sess.run([train_step, mean_batch_loss, logits],
                                      feed_dict={batchX_placeholder: audio_batch_vals_training,
                                                 batchY_placeholder: arr_2dims_to_1dim(label_batch_vals_training),
                                                 })
        if (iteration_idx + 1) % display_step == 0 or iteration_idx == 0:
            
            validation_iterations = 1 # 验证一个batch的训练数据
            cur_validation_acc = 0.
            for _ in range(validation_iterations):

                logits_validation, loss_val_validation = sess.run([logits, mean_batch_loss],
                                                                  feed_dict={batchX_placeholder: audio_batch_vals_training,
                                                                             batchY_placeholder: arr_2dims_to_1dim(label_batch_vals_training),# keep_prob: 1.0
                                                                             })
                validation_accuracy = get_accuracy(label_batch_vals_training, logits_validation)
                cur_validation_acc += validation_accuracy

            cur_validation_acc /= validation_iterations
            time_end = time.time()
            print("iter %d, training loss: %f, validation accuracy: %f, lr= %f, time=%f sec" % ((iteration_idx + 1), loss_val, cur_validation_acc, sess.run(learning_rate), time_end-time_start))
            time_start = time.time()
    if not os.path.exists(os.path.join(args.pathOfScratchFolder,'preprocessing','model')):
        os.mkdir(os.path.join(args.pathOfScratchFolder,'preprocessing','model'))
    save_path = saver.save(sess, os.path.join(args.pathOfScratchFolder,'preprocessing','model','model_512_5l_LSTM.ckpt'))
    print("#########      Training finish and model has been saved.    #########")
    print('model path: %s' % save_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()
