from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import os
import sys
import numpy as np

import train
import dataset


data1_index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
data2_index = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
DATA1_DIR = "./data/mnist/"
DATA2_DIR = "./data/emnist/"
MODEL1_DIR = "./model1/"
MODEL2_DIR = "./model2/"
MODEL_DIR = "./model/"
MODEL_EVAL_DIR = "./model1/"
NUM_TRAIN = 45000
NUM_TEST = 8000

num_sample = 100
num_train_sample = 80
num_eval_sample = 10

NUM_EPOCHS = 1000
LOG_EPOCHS = 10
LEARNING_RATE = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='generate',
                    help='Either `generate`, or `classify`, or `evalute`')

def generate():
    data1 = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data1_index)
    data2 = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data2_index)

    for i in range(num_sample):
        print("---------- Iteration " + str(i) + " ----------")
        train.Train(MODEL1_DIR + str(i), data1)
        train.Train(MODEL2_DIR + str(i), data2)

def classify():
    params1_list = []
    params2_list = []
    for i in range(num_sample):
        tf.reset_default_graph()
        g = tf.Graph()
        with tf.Session(graph = g) as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL1_DIR + str(i))
            g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            g_saver.restore(sess, ckpt.model_checkpoint_path)

            _list = []
            for var in tf.trainable_variables():
                result = sess.run(var)
                _list.append(np.average(result))
                _list.append(np.std(result))
            params1_list.append(_list)

    for i in range(num_sample):
        tf.reset_default_graph()
        g = tf.Graph()
        with tf.Session(graph = g) as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL2_DIR + str(i))
            g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            g_saver.restore(sess, ckpt.model_checkpoint_path)

            _list = []
            for var in tf.trainable_variables():
                result = sess.run(var)
                _list.append(np.average(result))
                _list.append(np.std(result))
            params2_list.append(_list)

    params1 = np.stack(params1_list)
    params2 = np.stack(params2_list)

    train_x = np.concatenate((params1[:num_train_sample],params2[:num_train_sample]), axis=0)
    train_y = [[1, 0]]*num_train_sample + [[0, 1]]*num_train_sample
    validate_x = np.concatenate((params1[num_train_sample:],params2[num_train_sample:]), axis=0)
    validate_y = [[1, 0]]*(num_sample-num_train_sample)+[[0, 1]]*(num_sample-num_train_sample)

    dim = params1.shape[1]
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, dim], name="X")
    y = tf.placeholder(tf.float32, [None, 2], name="Y")
    logits = tf.layers.dense(inputs=x, units=2, name="logits")
    soft = tf.nn.softmax(logits, name="soft")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits, labels=y), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name="optimizer").minimize(cost)

    correct_pred = tf.equal(tf.argmax(soft, 1), tf.argmax(y, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for step in range(NUM_EPOCHS):
            feed_dict = {x: train_x, y: train_y}
            cost_train, acc_train, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)

            if step % LOG_EPOCHS == 0:
                feed_dict = {x: validate_x, y: validate_y}
                _, acc_val = sess.run([cost, accuracy], feed_dict=feed_dict)

                print('[%d]: loss = %.4f train_acc = %.4f validate_acc = %.4f'
                      % (step, cost_train, acc_train, acc_val))


        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        saver.save(sess, MODEL_DIR+"model.ckpt")


def evaluate():
    params_list = []
    for i in range(num_eval_sample):
        tf.reset_default_graph()
        g = tf.Graph()
        with tf.Session(graph = g) as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_EVAL_DIR + str(i))
            g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            g_saver.restore(sess, ckpt.model_checkpoint_path)

            var = [v for v in tf.trainable_variables() if v.name == params_name][0]
            result = sess.run(var)
            params_list.append(result.flatten())

    eval_x = np.stack(params_list)
    eval_y = [[1, 0]]*num_eval_sample

    tf.reset_default_graph()
    g = tf.Graph()

    with tf.Session(graph = g) as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        g_saver.restore(sess, ckpt.model_checkpoint_path)

        x = g.get_tensor_by_name("X:0")
        y = g.get_tensor_by_name("Y:0")
        cost = g.get_tensor_by_name("cost:0")
        accuracy = g.get_tensor_by_name("accuracy:0")

        feed_dict = {x: eval_x, y: eval_y}
        _, acc_test = sess.run([cost, accuracy], feed_dict=feed_dict)

        print('test_acc = %.4f' % (acc_test))


def main(argv=None):

    FLAGS = parser.parse_args()
    if (FLAGS.mode == "generate"):
        generate()
    elif (FLAGS.mode == "classify"):
        classify()
    elif (FLAGS.mode == "evaluate"):
        evaluate()
    else:
        raise ValueError("set --mode as 'generate', 'classify' or 'evaluate'")


if __name__ == "__main__":
    main()
