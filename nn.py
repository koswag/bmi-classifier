import numpy as np
import random

_learn_rate = 0.01


def set_learn_rate(lr):
    global _learn_rate
    _learn_rate = lr


class Model:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return softmax(x)

    def backward(self, y, d):
        out = self.layers[-1]
        delta = d - y
        out.update(delta)

        prev = out
        for layer in reversed(self.layers[:-1]):
            delta = np.dot(prev.weights.T, delta)
            layer.update(delta)
            prev = layer

    def __getitem__(self, i):
        return self.layers[i]

    def __call__(self, x):
        x = np.array(x)
        return self.forward(x)


class Linear:
    def __init__(self, in_count, out_count, act):
        self.weights = np.random.rand(out_count, in_count)
        self.b = np.random.rand(out_count)
        self.act = act

    def update(self, delta):
        err = self.deriv(self.net) * delta
        self.weights += np.outer(err, self.x) * _learn_rate
        self.b += err * _learn_rate

    def deriv(self, x):
        if self.act == sigmoid:
            return sig_deriv(x)
        elif self.act == relu:
            return relu_deriv(x)

    def __call__(self, x):
        self.x = np.array(x)
        net = np.dot(self.weights, x) + self.b
        self.net = net
        return self.act(net)


class Classifier:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return response(self.model(x))

    def train(self, train_data, test_data, target_acc):
        e = 1
        prev_acc = 0
        accuracy = 0
        while accuracy < target_acc:
            # training step
            correct = 0
            random.shuffle(train_data)
            for row in train_data:
                x = np.array(row[0])
                d = np.array(row[1])
                y = self.model(x)
                if validate(response(y), d):
                    correct += 1
                self.model.backward(y, d)

            train_acc = correct/len(train_data)

            # validation step
            correct = 0
            for row in test_data:
                x = np.array(row[0])
                d = np.array(row[1])
                res = response(self.model(x))
                if validate(res, d):
                    correct += 1

            accuracy = correct / len(test_data)
            if accuracy != prev_acc:
                print(f'Epoch {e}: \n\tTrain: {train_acc * 100}\n\tTest: {accuracy * 100}')
            prev_acc = accuracy

            e += 1
        return accuracy, e


def validate(y, d):
    if len(y) == len(d):
        for i in range(len(y)):
            if y[i] != d[i]:
                return False
        return True


def response(output):
    res = max(output)
    return [1 if x == res else 0 for x in output]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))


def relu(x):
    return x * (x > 0)


def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def io_count(data):
    i = len(data[0][0])
    o = len(data[0][1])
    return i, o
