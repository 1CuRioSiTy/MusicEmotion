import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

a = tf.placeholder(tf.float32, [2, 2, 2])
y = tf.unstack(a, axis=0)
print(a.get_shape())
with tf.Session() as sess:
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(sess.run(y, feed_dict={a: array}))






