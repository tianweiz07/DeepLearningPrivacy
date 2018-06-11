from sklearn.metrics import classification_report, accuracy_score
import theano.tensor as T
import numpy as np
import lasagne
import theano
import argparse
from load_cifar import load_cifar

def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def get_nn_model(n_in, n_hidden, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


def get_softmax_model(n_in, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['output'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_cnn_model(n_in, n_hidden, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, 3, 32, 32))
    net['conv1'] = lasagne.layers.Conv2DLayer(
            net['input'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['pool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    net['fc'] = lasagne.layers.DenseLayer(
            net['pool2'],
            num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.tanh)
    net['output'] = lasagne.layers.DenseLayer(
            net['fc'],
            num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax)
    return net


def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7, rtn_layer=True):
    train_x, train_y, test_x, test_y = dataset

    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    print 'Building model with {} training data, {} classes...'.format(len(train_x), n_out)
    if model == 'cnn':
        train_x = np.dstack((train_x[:, :1024], train_x[:, 1024:2048], train_x[:, 2048:]))
        train_x = train_x.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
        test_x = np.dstack((test_x[:, :1024], test_x[:, 1024:2048], test_x[:, 2048:]))
        test_x = test_x.reshape((-1, 32, 32, 3)).transpose(0, 3, 1, 2)
        input_var = T.tensor4('x')
    else:
        input_var = T.matrix('x')
    target_var = T.ivector('y')

    if model == 'nn':
        print 'Using neural network...'
        net = get_nn_model(n_in, n_hidden, n_out)
    elif model == 'softmax':
        print 'Using softmax regression...'
        net = get_softmax_model(n_in, n_out)
    else:
        print 'Using convolutional neural network...'
        net = get_cnn_model(n_in, n_hidden, n_out)

    net['input'].input_var = input_var
    output_layer = net['output']

    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)

    print 'Training...'
    for epoch in range(epochs):
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)
        loss = round(loss, 3)

        pred_y = []
        for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        train_acc = accuracy_score(train_y, pred_y)

        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        test_acc = accuracy_score(test_y, pred_y)
        print 'Epoch {}, train loss {}, Training Accuracy: {}, Testing Accuracy: {}'\
              .format(epoch, loss, train_acc, test_acc)

    if rtn_layer:
        return output_layer
    else:
        return pred_y


def load_dataset(data_dir):
    train_x, train_y, test_x, test_y = load_cifar(10, data_dir)
    return train_x, train_y, test_x, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print vars(args)
    dataset = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    train(dataset,
          model=args.model,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          n_hidden=args.n_hidden,
          epochs=args.epochs)


if __name__ == '__main__':
    main()
