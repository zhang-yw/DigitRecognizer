from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
from argparse import ArgumentParser

PIXEL_DEPTH = 255
batch_size = 2800
total_batch = 10

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model-dir', dest='model_directory', metavar='MODEL_DIRECTORY', required=True)
    return parser

def CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 28, 28, 1])
        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 1024, scope='fc')
        net = slim.dropout(net, is_training=is_training, scope='dropout')
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

def predict(model_directory):
    test = pd.read_csv('data/test.csv')
    test_data = numpy.float32(test.values)
    
    is_training = tf.placeholder(tf.bool, name='MODE')
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) 
    y = CNN(x, is_training=is_training)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    saver = tf.train.Saver()
    saver.restore(sess, model_directory)
    for i in range(total_batch):
        batch = test_data[i*batch_size:(i+1)*batch_size]
        batch_xs = (batch - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        if i != 0:
            y_final = numpy.concatenate((y_final, sess.run(y, feed_dict={x: batch_xs, is_training: False})), 0)
        else:
            y_final = sess.run(y, feed_dict={x: batch_xs, is_training: False})
    test_labels = numpy.argmax(y_final, axis=1)    
    result = pd.DataFrame(data={'ImageId':(numpy.arange(y_final.shape[0])+1), 'Label':test_labels})
    result.to_csv('submission.csv', index=False)
    print("Predicting Finished!")

if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    model_directory = options.model_directory

    predict(model_directory+'/model.ckpt')