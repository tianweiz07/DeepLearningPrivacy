from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import train

data1_index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
data2_index = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

num_sample = 100

param_list = []
for i in range(num_sample):
    print("data1: " + str(i))
    logit_weight = train.Train(data1_index)
    param_list.append(logit_weight)
params = np.stack(param_list)
np.save('param1.npy', params)
#x = np.load('result.npy')


param_list = []
for i in range(num_sample):
    print("data2: " + str(i))
    logit_weight = train.Train(data2_index)
    param_list.append(logit_weight)
params = np.stack(param_list)
np.save('param2.npy', params)
