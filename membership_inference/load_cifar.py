import numpy as np
import os
import random


def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch_cifar10(filename):
    batch = np.load(filename)
    data = batch['data'] / 255.0  # scale between [0, 1]
    labels = np.asarray(batch['labels'])  # convert labels to one-hot representation
    return data.astype('float32'), labels.astype('int32')


def _load_batch_cifar100(filename):
    batch = np.load(filename)
    data = batch['data'] / 255.0
    labels = np.asarray(batch['fine_labels'])
    return data.astype('float32'), labels.astype('int32')


def load_cifar10(data_dir):
    dir = data_dir
    train_x, train_y = [], []

    for filename in [1, 2, 3, 4, 5]:
        path = dir + 'data_batch_{}'.format(filename)
        x, y = _load_batch_cifar10(path)
        train_x.append(x)
        train_y.append(y)

    test_x, test_y = _load_batch_cifar10(dir + 'test_batch')

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    return train_x, train_y, test_x, test_y


def load_cifar100(data_dir):
    dir = data_dir
    train_x, train_y = _load_batch_cifar100(dir + 'train')
    test_x, test_y = _load_batch_cifar100(dir + 'test')
    return train_x, train_y, test_x, test_y


def load_cifar(label_size, data_dir):
    if label_size == 10:
        return load_cifar10(data_dir)
    elif label_size == 100:
        return load_cifar100()


if __name__ == '__main__':
    pass
