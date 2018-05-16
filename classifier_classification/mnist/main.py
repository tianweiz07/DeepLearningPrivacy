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
SAVE_EPOCHS = 200
LOG_EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_CLASS = 10

NUM_TRAIN = 45000
NUM_TEST = 8000

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
                    help='Either `train`, or `eval`.')

data_index = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2]

def Train():
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

    mnist = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data_index)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.all_variables())
            variables_to_restore = {}
            for v in tf.trainable_variables():
                variables_to_restore[v.op.name] = v

            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
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


            if step % SAVE_EPOCHS == 0:

                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                saver.save(sess, MODEL_DIR +"model.ckpt-"+str(step))

            step += 1

        saver.save(sess, MODEL_DIR +"model.ckpt-"+str(step))


def Eval():

    mnist = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data_index)

    g = tf.Graph()

    with tf.Session(graph = g) as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        g_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        g_saver.restore(sess, ckpt.model_checkpoint_path)

        x = g.get_tensor_by_name("X:0")
        y = g.get_tensor_by_name("Y:0")
        cost = g.get_tensor_by_name("cost:0")
        accuracy = g.get_tensor_by_name("accuracy:0")
        keep_rate = g.get_tensor_by_name("keep_rate:0")

        images_test = mnist.test.images
        labels_test = mnist.test.labels

        feed_dict = {x: images_test, y: labels_test, keep_rate: 1.0}
        _, acc_test = sess.run([cost, accuracy], feed_dict=feed_dict)

        print('test_acc = %.4f' % (acc_test))


def main(argv=None):

    FLAGS = parser.parse_args()
    if (FLAGS.mode == 'train'):
        Train()
    elif (FLAGS.mode == 'eval'):
        Eval()
    else:
        raise ValueError("set --mode as 'train' or 'eval'")


if __name__ == "__main__":
    tf.app.run()
