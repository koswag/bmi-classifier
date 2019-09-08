""" Utility module for processing and displaying data.
"""
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


class DataSet:
    """ Class representing a dataset of pairs (x, y)

    - x - input values

    - y - expected output value(s)
    """
    def __init__(self, entries):
        self.entries = list(entries)

    @staticmethod
    def from_io(inputs, outputs):
        entries = zip(inputs, outputs)
        return DataSet(entries)

    @staticmethod
    def empty():
        return DataSet([])

    @property
    def inputs(self):
        return col(self.entries, 0)

    @property
    def outputs(self):
        return col(self.entries, 1)

    def add(self, x, y):
        self.entries.append((x, y))

    def subsets(self, categories=None):
        if not categories:
            categories = unique(self.outputs)
        for c in categories:
            yield self.subset(c)

    def subset(self, category):
        filtered = [(x, y) for x, y in self.entries if y == category]
        return DataSet(filtered)

    def __getitem__(self, i):
        return self.entries[i]


def col(matrix, i):
    """ Get i-th column of a matrix. """
    return [row[i] for row in matrix]


def unique(arr):
    """ Get list of unique values from a list. """
    res = []
    for x in arr:
        if x not in res:
            res.append(x)
    return res


def split(data: DataSet, test_size=0.4, normalized=True):
    """ Split dataset to train and test set.

        :param data: DataSet to split
        :param test_size: Desired test size between 0 and 1.
        :param normalized: Bool indicating whether inputs should be normalized.

        :return: Train and test set as a tuple
    """
    if not 0 < test_size < 1:
        raise ValueError('Test size has to be between 0 and 1')

    x, y = data.inputs, data.outputs
    in_train, in_test, out_train, out_test = train_test_split(x, y, test_size=test_size)

    if normalized:
        in_train, in_test = normalize(in_train), normalize(in_test)

    train_data = DataSet.from_io(in_train, out_train)
    test_data = DataSet.from_io(in_test, out_test)
    return train_data, test_data


def get_path(relative_path):
    script_dir = os.path.dirname(__file__)
    absolute = os.path.abspath(script_dir)
    return os.path.join(absolute, relative_path)


_colors = ('b', 'g', 'r', 'c',
           'm', 'y', 'k', 'w')


def plot(data: DataSet, categories=None, title=None, show=False, num=1, n_rows=1, n_cols=1):
    """ Plot 2-dimensional data.

        :param data: DataSet with 2D input.
        :param categories: Complete list of possible output categories
        :param title: Figure title.
        :param show: Bool indicating whether plot is to be shown.
        :param num: Subplot's positional number on the grid.
        :param n_rows: Number of rows in the grid.
        :param n_cols: Number of columns in the grid.
    """
    plt.subplot(n_rows, n_cols, num)

    i = 0
    for sub in data.subsets(categories):
        x, y, = col(sub.inputs, 0), col(sub.inputs, 1)
        plt.plot(x, y, f'{_colors[i]}o')
        i += 1

    if title:
        plt.title(title)

    if show:
        plt.show()
