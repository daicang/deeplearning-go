import random

class MSE:

    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff * diff)[0]

    @staticmethod
    def loss_derivative(predictions, labels):
        return predictions - labels


class SequentialNetwork:
    def __init__(self, loss=None):
        print("Initializing SequentialNetwork")
        self.layers = []
        self.loss = loss or MSE()

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for batch in mini_batches:
                self.train_batch(batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                print(f'Epoch {i}: {self.evaluate(test_data)}/{n_test}')
            else:
                print(f'Epoch {i} complete')

    def train_batch(self, batch, learning_rate):
        self.forward_backward(batch)
        self.update(batch, learning_rate)

    def update(self, batch, rate):
        rate = rate / len(batch)
        for layer in self.layers:
            layer.update_params(rate)
            layer.clear_deltas()
