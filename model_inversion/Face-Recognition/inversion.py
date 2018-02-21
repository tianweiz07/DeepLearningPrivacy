import time
import math
import os
import data
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

NEpochs = 100
learningRate = 1e-1

x = tf.Variable(tf.zeros([1, imageSize], dtype=tf.float32))
y = 20
w = tf.placeholder(tf.float32, shape=(imageSize, NClasses))
b = tf.placeholder(tf.float32, shape=(NClasses))

#X, Y= data.LoadTrainingData(training_dir, (imageWidth, imageHeight))
#for i in range(360):
#    if Y[i][y] == 1:
#        break
#init = tf.constant(X.reshape([-1, imageSize])[i].reshape([1, imageSize]))
#x = tf.Variable(init)

def main():
    with tf.Session() as sess:
        # initialize variable
        sess.run(tf.global_variables_initializer())
        # restore model
        new_saver = tf.train.import_meta_graph(model_dir+'model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.get_default_graph()
        weights = sess.run(graph.get_tensor_by_name("weights:0"))
        biases = sess.run(graph.get_tensor_by_name("biases:0"))

        # define loss function and optimizer
        prediction = tf.nn.softmax(tf.matmul(x, w) + b)
        loss =  1 - prediction[0][y]
        optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
        for epoch in range(NEpochs):
            _, l = sess.run([optimizer, loss], feed_dict={w: weights, b: biases})
            if epoch % 1 == 0:

                data = np.asarray(sess.run(x), dtype = np.float32).reshape([-1, imageHeight, imageWidth])[0]
                data_rescaled = (255.0 / data.max() * data).astype(np.uint8)

                plt.imshow(np.asarray(sess.run(x)).reshape([-1, imageHeight, imageWidth])[0], cmap='gray')
                plt.savefig('inversion/test_'+str(epoch)+'.png')

                print epoch, l

        # save image
        plt.imshow(np.asarray(sess.run(x)).reshape([-1, imageHeight, imageWidth])[0], cmap='gray')
        plt.show()

        data = np.asarray(sess.run(x), dtype=np.float32).reshape([-1, imageHeight, imageWidth])[0]
        img = Image.fromarray(data, 'L')
        img.save('test.png')           

if __name__ == '__main__':
    main()
