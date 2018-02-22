import time
import math
import os
import data
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_dir = 'Datasets/Training'
testing_dir = 'Datasets/Testing'
model_dir = 'checkpoints/'

imageWidth = 92
imageHeight = 112
imageSize = imageWidth * imageHeight
NChannels = 1
NClasses = 40

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='# of Epochs')
    parser.add_argument('--learn', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--photo', type=int, default=1, help='photo index')

    FLAGS = parser.parse_args()
    NEpochs = FLAGS.epoch
    learningRate = FLAGS.learn

    x = tf.Variable(tf.zeros([1, imageSize], dtype=tf.float32))
    y = FLAGS.photo
    w = tf.placeholder(tf.float32, shape=(imageSize, NClasses))
    b = tf.placeholder(tf.float32, shape=(NClasses))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph(model_dir+'model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.get_default_graph()
        weights = sess.run(graph.get_tensor_by_name("weights:0"))
        biases = sess.run(graph.get_tensor_by_name("biases:0"))

        prediction = tf.nn.softmax(tf.matmul(x, w) + b)
        loss =  1 - prediction[0][y]
        optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
        for epoch in range(NEpochs):
            _, l = sess.run([optimizer, loss], feed_dict={w: weights, b: biases})
            print("Epoch: %01d loss: %.4f" % (epoch+1, l))

        data = np.asarray(sess.run(x), dtype = np.float32).\
                          reshape([-1, imageHeight, imageWidth])[0]
        data_rescaled = (255.0 / data.max() * data).astype(np.uint8)
        plt.imshow(np.asarray(sess.run(x)).reshape([-1, imageHeight, imageWidth])[0],cmap='gray')
#        plt.savefig('inversion/test_'+str(y)+'.png')
        plt.savefig('/home/ubuntu/test_'+str(y)+'.png')

if __name__ == '__main__':
    main()
