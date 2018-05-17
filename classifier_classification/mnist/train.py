from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import sys
import numpy as np

import network

tf.logging.set_verbosity(tf.logging.INFO)

NUM_EPOCHS = 500
LOG_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_CLASS = 10


def Train(MODEL_DIR, data):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
    y = tf.placeholder(tf.int32, [None, NUM_CLASS], name="Y")
    keep_rate = tf.placeholder(tf.float32, name="keep_rate")

    logits = network.CreateNet(x, keep_rate)

    soft = tf.nn.softmax(logits, name="soft")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, labels=y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name="optimizer").minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(soft, 1), tf.argmax(y, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        global_step = 0

        step = global_step + 1

        while step < global_step + NUM_EPOCHS:
            batch_x, batch_y = data.train.next_batch(BATCH_SIZE)
            feed_dict = {x: batch_x, y: batch_y, keep_rate: 0.8}
            cost_train, acc_train, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)

            if step % LOG_EPOCHS == 0:
                images_val = data.test.images
                labels_val = data.test.labels
                feed_dict = {x: images_val, y: labels_val, keep_rate: 1.0}
                _, acc_val = sess.run([cost, accuracy], feed_dict=feed_dict)

                print('[%d]: loss = %.4f train_acc = %.4f validate_acc = %.4f'
                      % (step, cost_train, acc_train, acc_val))

            step += 1

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        saver.save(sess, MODEL_DIR+"/model.ckpt")
#        var = [v for v in tf.trainable_variables() if v.name == "logits/kernel:0"][0]
#        result = sess.run(var)

#        return result
