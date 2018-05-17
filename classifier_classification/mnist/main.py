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
NUM_TRAIN = 45000
NUM_TEST = 8000


num_sample = 100

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='generate',
                    help='Either `generate`, or `classify`,')

def generate():
    data1 = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data1_index)
    data2 = dataset.read_data_sets(DATA1_DIR, DATA2_DIR, reshape=False, one_hot=True,
                                   num_train=NUM_TRAIN, num_test=NUM_TEST, data_index = data2_index)

    for i in range(num_sample):
        print("---------- Iteration " + str(i) + " ----------")
        train.Train("model1/" + str(i), data1)
        train.Train("model2/" + str(i), data2)


def main(argv=None):

    FLAGS = parser.parse_args()
    if (FLAGS.mode == "generate"):
        generate()
    elif (FLAGS.mode == "classify"):
        classify()
    else:
        raise ValueError("set --mode as 'generate' or 'classify'")


if __name__ == "__main__":
    main()

#param_list = []
#for i in range(num_sample):
#    print("data1: " + str(i))
#    logit_weight = train.Train(data1_index)
#    param_list.append(logit_weight)
#params = np.stack(param_list)
#np.save('param1.npy', params)
#x = np.load('result.npy')


#param_list = []
##    print("data2: " + str(i))
#    logit_weight = train.Train(data2_index)
#    param_list.append(logit_weight)
#params = np.stack(param_list)
#np.save('param2.npy', params)
