import numpy as np


def sigmoid_double(i):
    return 1.0 / (1.0 + np.exp(-i))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)


def sigmoid_prime_double(i):
    sd = sigmoid_double(i)
    return sd * (1 - sd)


def sigmoid_price(z):
    return np.vectorize(sigmoid_double)(z)


class Layer:
    def __init__(self):
        self.params = []
        self.previous = None
        self.next = None

        self.input_data = None
        self.output_data = None
        self.input_delta = None
        self.output_delta = None

    def connect(self, layer: "Layer"):
        self.next = layer
        layer.previous = self

    def forward(self):
        raise NotImplementedError

    def get_forward_input(self):
        if self.previous:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        raise NotImplementedError

    def get_backward_input(self):
        if self.next:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(self):
        pass

    def update_params(self, learning_rate):
        pass

    def describe(self):
        raise NotImplementedError
