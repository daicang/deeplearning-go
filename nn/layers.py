import numpy as np


def sigmoid(i):
    return 1.0 / (1.0 + np.exp(-i))


def sigmoid_prime(i):
    sd = sigmoid(i)
    return sd * (1 - sd)


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


class ActivationLayer(Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        self.input_data = self.get_forward_input()
        self.output_data = sigmoid(self.input_data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_data = delta * sigmoid_prime(data)

    def describe(self):
        return f"ActivationLayer(dim={self.input_dim, self.output_dim})"


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()

        self.delta_b += delta
        self.delta_w += np.dot(delta, data.T)

        self.output_delta = np.dot(self.weight.T, delta)

    def update_params(self, rate):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        return f"DenseLayer(input_dim={self.input_dim}, output_dim={self.output_dim})"
