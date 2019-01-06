from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import mnist_data

MODEL_DIRECTORY = "model/model.ckpt"
training_epochs = 10
batch_size = 50
print_step = 100
validation_step = 500
num_labels = 10

def CNN(inputs, is_training=True):
    normlize_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=normalize_params):
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

def train():
    train_total_data, train_size, validation_data, validation_labels = mnist_data.prepare_MNIST_data(True)

    is_training = tf.placeholder(tf.bool, name='MODE')
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) 
    y = CNN(x)
    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y,y_)
    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1e-4, batch * batch_size, train_size, 0.95, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)
    with tf.name_scope("ACCR"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    total_batch = int(train_size / batch_size)
    max_acc = 0.
    for epoch in range(training_epochs):
        numpy.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-num_labels]
        train_labels_ = train_total_data[:, -num_labels:]
        for i in range(total_batch):
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            _, train_accuracy = sess.run([train_step, accuracy] , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            if i % print_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1), "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

            if i % validation_step == 0:
                validation_accuracy = sess.run(accuracy, feed_dict={x: validation_data, y_: validation_labels, is_training: False})
                print("Epoch:", '%04d,' % (epoch + 1), "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY)
                print("Model updated and saved in file: %s" % save_path)
    print("Training Finished!")

if __name__ == '__main__':
    train()
