from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import numpy
from six.moves import xrange 
from random import randint

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    seed1, seed2 = random_seed.get_seed(seed)
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = images.shape[0]

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def read_data_sets(data1_dir,
                   data2_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   seed=None,
                   num_train=0,
                   num_test=0,
                   data_index=None,
                   noise=0):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    test = fake()
    return base.Datasets(train=train, validation=None, test=test)

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = os.path.join(data1_dir, TRAIN_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    train1_labels = extract_labels(f, one_hot=False)

  local_file = os.path.join(data1_dir, TRAIN_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    train1_images = extract_images(f)

  local_file = os.path.join(data1_dir, TEST_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    test1_labels = extract_labels(f, one_hot=False)

  local_file = os.path.join(data1_dir, TEST_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    test1_images = extract_images(f)

  local_file = os.path.join(data2_dir, TRAIN_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    train2_labels = extract_labels(f, one_hot=False)

  local_file = os.path.join(data2_dir, TRAIN_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    train2_images = extract_images(f)

  local_file = os.path.join(data2_dir, TEST_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    test2_labels = extract_labels(f, one_hot=False)

  local_file = os.path.join(data2_dir, TEST_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    test2_images = extract_images(f)

  train1_index = []
  train2_index = []
  test1_index = []
  test2_index = []

  for digit in range(10):
    if data_index[digit] == 1:
      num_index = 0
      for i in range(train1_labels.shape[0]):
        if train1_labels[i] == digit:
          train1_index.append(i)
          num_index += 1
        if num_index == num_train/10:
          break
    elif data_index[digit] == 2:
      num_index = 0
      for i in range(train2_labels.shape[0]):
        if train2_labels[i] == digit:
          train2_index.append(i)
          num_index += 1
        if num_index == num_train/10:
          break

    if data_index[digit] == 1:
      num_index = 0
      for i in range(test1_labels.shape[0]):
        if test1_labels[i] == digit:
          test1_index.append(i)
          num_index += 1
        if num_index == num_test/10:
          break
    elif data_index[digit] == 2:
      num_index = 0
      for i in range(test2_labels.shape[0]):
        if test2_labels[i] == digit:
          test2_index.append(i)
          num_index += 1
        if num_index == num_test/10:
          break
    
  train_images = numpy.concatenate((train1_images[train1_index], 
                                    train2_images[train2_index]), axis=0)
  train_labels = numpy.concatenate((train1_labels[train1_index],
                                    train2_labels[train2_index]), axis=0)
  test_images = numpy.concatenate((test1_images[test1_index],
                                   test2_images[test2_index]), axis=0)
  test_labels = numpy.concatenate((test1_labels[test1_index],
                                   test2_labels[test2_index]), axis=0)


  # ---- Add random noise----------
  (height, width, channel) = train_images[0].shape
  for i in range(int(noise*num_train)):
    _index = randint(0, num_train-1)
    _image = train_images[_index]
    new_image = numpy.expand_dims(numpy.clip(255 - _image - numpy.random.normal(scale=30,
                                  size=(height, width, channel)), 0, 255), axis=0)
    new_label = numpy.expand_dims(train_labels[_index], axis=0)
    train_images = numpy.vstack((train_images, new_image))
    train_labels = numpy.concatenate((train_labels, new_label), axis=0)
  # ------------------------------

  train_index = range(train_labels.shape[0])
  test_index = range(test_labels.shape[0])

  numpy.random.shuffle(train_index)
  numpy.random.shuffle(test_index)

  train_images = train_images[train_index]
  train_labels = train_labels[train_index]
  test_images = test_images[test_index]
  test_labels = test_labels[test_index]

  if one_hot:
    train_labels = dense_to_one_hot(train_labels, 10)
    test_labels = dense_to_one_hot(test_labels, 10)
  
  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=None, test=test)
