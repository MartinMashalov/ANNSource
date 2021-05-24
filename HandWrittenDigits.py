import numpy as np
import math
import random
import mnist_loader

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

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def _sigmoid(z):
        return 1/(1 + np.exp(-z))

    @staticmethod
    def _cost_derivative(output_activations, y):
        return output_activations - y

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    #a is the input "x" used before
    #return the output of the network if the input was "a" vector
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(np.dot(w, a) + b)
        return a

    #heuristic for measuring accuracy of neural network output
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #training data is just a bunch of tuple (input, output)
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=False):

        #if test_data is True:
        length_test = len(test_data)
        length_training = len(training_data)

        #epochs are the number of steps to take to find the minimum
        #learning_rate is n
        for epoch_index in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, length_training, mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, learning_rate)

            if test_data:
                print("Epoch {}: {} / {}".format(epoch_index, self.evaluate(test_data), length_test))
            else:
                print("Epoch {} complete".format(epoch_index))

    #apply a single step of gradient descent over a mini batch
    def update_mini_batch(self, mini_batch, epoch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nab + ban for nab, ban in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nab + ban for nab, ban in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(epoch/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(epoch/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self._cost_derivative(activations[-1], y) * self._sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.layers):
            z = zs[-l]
            sp = self._sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w


training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()
network = NeuralNetwork([784, 30, 10])
network.stochastic_gradient_descent(training_data, 60, 10, 3.0, test_data=testing_data)



