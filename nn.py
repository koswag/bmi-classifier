import numpy as np
import random


class Linear:
    """ Class representing a fully connected linear layer. """
    def __init__(self, in_count, out_count, act_fun):
        self.weights = np.random.rand(out_count, in_count)
        self.b = np.random.rand(out_count)
        self.act = act_fun

        self.x = np.zeros(in_count)
        self.net = np.zeros(out_count)

    def output(self, x):
        self.x = np.array(x)
        net = self.weights @ self.x + self.b
        self.net = net
        return self.act(net)

    def update(self, delta, learn_rate):
        err = self.deriv(self.net) * delta
        self.weights += np.outer(err, self.x) * learn_rate
        self.b += err * learn_rate

    def deriv(self, x):
        if self.act == sigmoid:
            return sig_deriv(x)
        elif self.act == relu:
            return relu_deriv(x)
        elif self.act == tanh:
            return tanh_deriv(x)

    def __call__(self, x):
        return self.output(x)


class Model:
    def __init__(self, *layers, learn_rate=0.01):
        self.layers = layers
        self.lr = learn_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return softmax(x)

    def backward(self, y, d):
        y, d = np.array(y), np.array(d)

        out_layer = self.layers[-1]
        delta = d - y
        out_layer.update(delta, self.lr)

        prev = out_layer
        for layer in reversed(self.layers[:-1]):
            delta = np.dot(prev.weights.T, delta)
            layer.update(delta, self.lr)
            prev = layer

    def __getitem__(self, i):
        return self.layers[i]

    def __call__(self, x):
        return self.forward(x)


class Classifier:
    def __init__(self, model):
        self.model = model

    def output(self, x):
        return response(self.model(x))

    def train(self, train_data, test_data, target_acc):
        train_data, test_data = list(train_data.entries), list(test_data.entries)

        prev_acc = 0
        test_accuracy = 0

        e = 1
        while test_accuracy < target_acc:
            train_accuracy = self.train_step(train_data)
            test_accuracy = self.validation_step(test_data)

            if test_accuracy != prev_acc:
                print(f'Epoch {e}: \n\tTrain: {(train_accuracy * 100):.2f}\n\tTest: {(test_accuracy * 100):.2f}')
            prev_acc = test_accuracy

            e += 1
        return test_accuracy, e

    def train_step(self, train_data):
        n_correct = 0
        random.shuffle(train_data)

        for row in train_data:
            x, d = row
            y = self(x)

            if validate(y, d):
                n_correct += 1
            self.model.backward(y, d)

        return n_correct / len(train_data)

    def validation_step(self, test_data):
        n_correct = 0

        for row in test_data:
            x, d = row
            y = self(x)

            if validate(y, d):
                n_correct += 1

        return n_correct / len(test_data)

    def __call__(self, x):
        return self.output(x)


def response(output):
    res = max(output)
    return [int(x == res) for x in output]


def validate(y, d):
    if len(y) == len(d):
        for i in range(len(y)):
            if y[i] != d[i]:
                return False
        return True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_deriv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return x * (x > 0)


def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - tanh(x)**2


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
