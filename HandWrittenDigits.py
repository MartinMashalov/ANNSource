import numpy as np
import math


class NeuralNetwork:

    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []

        def create_biases():
            for node in sizes[1:]:
                self.biases.append(np.random.rand(node, 1))
            self.biases = np.array(self.biases)

        #TODO: create my own custom weights system
        def create_weights():
            for x, y in zip(sizes[:-1], sizes[:1]):
                self.weights.append(np.random.rand(x, y))
            self.weights = np.array(self.weights)

        create_biases()
        create_weights()

        #print(sizes)

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def _sigmoid(z):
        return 1/(1 + np.exp(-z))

    #a is the input "x" used before
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(np.dot(w, a) + b)

        return a

    #training data is just a bunch of tuple (input, output)
    def stochastic_gradient_descent(self, epochs, training_data, mini_batch_size, eta, training_exists=False):

        if training_exists is True:
            pass

Net = NeuralNetwork([2, 4, 1])
print(Net.feedforward([]))
