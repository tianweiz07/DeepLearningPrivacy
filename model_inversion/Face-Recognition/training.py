import time
import math
import os
import data
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_dir = 'Datasets/Training'
testing_dir = 'Datasets/Testing'
model_dir = 'checkpoints/'

imageWidth = 92
imageHeight = 112
imageSize = imageWidth*imageHeight
NChannels = 1
NClasses = 40

BatchSize = 1
NEpochs = 400
learningRate = 1e-3

x = tf.placeholder(tf.float32, shape=(None, imageSize))
y = tf.placeholder(tf.float32, shape=(None, NClasses))

X, Y = data.LoadTrainingData(training_dir, (imageWidth, imageHeight))
data.TrainingData = X
data.TrainingLables = Y

XT, YT, NamesT, _, Paths = data.LoadTestingData(testing_dir, (imageWidth, imageHeight))
data.TestingData = XT
data.TestingLables = YT

#weights = tf.Variable(tf.truncated_normal([imageSize, NClasses]), name='weights')
weights = tf.get_variable(shape=[imageSize, NClasses], initializer=tf.contrib.layers.xavier_initializer(),name='weights')
biases = tf.Variable(tf.zeros([NClasses]), name='biases')

def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def main():
    logits = tf.matmul(x, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          labels=y, logits=logits))
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rang = int(len(X)/BatchSize)
 
        for epoch in range(NEpochs):
            avg_loss = 0.0
            avg_acc = 0.0
            for i in range(rang):
                epochX, epochY = data.nextBatch(BatchSize)
                epochX = epochX.reshape(len(epochX), imageSize)
                feed_dict = {x: epochX, y:epochY}
 
                _, l, predictions = sess.run([optimizer, loss, train_prediction],
                                                feed_dict=feed_dict)
                avg_acc += accuracy(predictions, epochY) /rang
                avg_loss += (l / (rang))
            print("Epoch: %01d/%01d loss: %.4f Accuracy: %.2f" % 
                  (epoch+1, NEpochs, avg_loss, avg_acc) + str(' %'))


            XT_input = XT.reshape(len(XT), imageSize)
            feed_dict = {x: XT_input, y:YT}
            test_predictions = sess.run(train_prediction,
                                                feed_dict=feed_dict)
            acc = accuracy(test_predictions, np.asarray(YT))
            print("Testing Accuracy: " + str(acc) + str(' %'))

        saver.save(sess, model_dir+'model')

if __name__ == '__main__':
    main()
