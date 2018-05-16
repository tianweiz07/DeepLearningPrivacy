from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

import utils
import dataset

mnist = dataset.read_data_sets("./data/mnist", "./data/emnist", reshape=False, one_hot=True, num_train=100, num_test=40, data_index=[1, 1, 2, 2, 1, 1, 2, 2, 1, 2])

#train_data = mnist.train.images
#train_labels = mnist.train.labels

#train_data, train_labels = utils.LoadTrain()

#print(train_data.shape)
#print(train_labels.shape)

#for i in range(50):
#    print(train_labels[i])
#    utils.Convert2Image(train_data[i], "/home/ubuntu/image1/" + str(i) + ".png")

#l = [0 for i in range(0, 100)]

#for x in train_labels:
#    l[x] += 1

#print(l)
