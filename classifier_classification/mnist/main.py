from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import os
import sys
import numpy as np

import train

data_index = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2]

logit_weight = train.Train(data_index)
print(logit_weight)

np.save('result.npy', logit_weight)
x = np.load('result.npy')

print(x)
