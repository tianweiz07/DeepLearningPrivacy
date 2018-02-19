import time
import math
import os
import data
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

testing_dir = 'Datasets/Testing'
model_dir = 'checkpoints/'

imageWidth = 92
imageHeight = 112
imageSize =imageWidth*imageHeight
NChannels = 1
NClasses = 40

XT, YT, NamesT, _, Paths = data.LoadTestingData(testing_dir, (imageWidth, imageHeight))
data.TestingData = XT
data.TestingLables = YT

XT = XT.reshape(len(XT), imageSize)

def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def main():
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_dir+'model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.get_default_graph()
        weights = graph.get_tensor_by_name("weights:0")
        biases = graph.get_tensor_by_name("biases:0")

        test_prediction = tf.nn.softmax(tf.matmul(XT, weights) + biases)
        acc = accuracy(test_prediction.eval(), np.asarray(YT))
        print("Testing Accuracy: " + str(acc) + str(' %'))

if __name__ == '__main__':
    main()
