import numpy as np


def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    labels = [encode_label(y) for y in data[1]]
    return list(zip(features, labels))


def load_data():
    f = np.load('mnist.npz')
    x_train, x_test = f['x_train'], f['x_test']
    y_train, y_test = f['y_train'], f['y_test']
    f.close()
    return shape_data((x_train, y_train)), shape_data((x_test, y_test))
