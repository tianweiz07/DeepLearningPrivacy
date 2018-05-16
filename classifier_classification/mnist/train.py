from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import sys
import numpy as np

import network
import dataset

tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIR = "./checkpoints/"
DATA1_DIR = "./data/mnist/"
DATA2_DIR = "./data/emnist/"
NUM_EPOCHS = 500
LOG_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_CLASS = 10

NUM_TRAIN = 45000
NUM_TEST = 8000

data_index = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2]

def Train(data_index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
    y = tf.placeholder(tf.int32, [None, NUM_CLASS], name="Y")
    keep_rate = tf.placeholder(tf.float32, name="keep_rate")

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        logits = network.CreateNet(x, keep_rate)

    soft = tf.nn.softmax(logits, name="soft")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, labels=y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name="optimizer").minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(soft, 1), tf.argmax(y, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    mnist = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data_index)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        global_step = 0

        step = global_step + 1

        while step < global_step + NUM_EPOCHS:
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            feed_dict = {x: batch_x, y: batch_y, keep_rate: 0.8}
            cost_train, acc_train, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)

            if step % LOG_EPOCHS == 0:
                images_val = mnist.test.images
                labels_val = mnist.test.labels
                feed_dict = {x: images_val, y: labels_val, keep_rate: 1.0}
                _, acc_val = sess.run([cost, accuracy], feed_dict=feed_dict)

                print('[%d]: loss = %.4f train_acc = %.4f validate_acc = %.4f'
                      % (step, cost_train, acc_train, acc_val))

            step += 1

        var = [v for v in tf.trainable_variables() if v.name == "logits/kernel:0"][0]
        result = sess.run(var)

        return result
