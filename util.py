import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def split(data, test_size):
    """Split dataset to train and test set."""
    in_train, in_test, out_train, out_test = train_test_split(col(data, 0), col(data, 1), test_size=test_size)
    in_train, in_test = normalize(in_train), normalize(in_test)

    train_data = [(in_train[i], out_train[i]) for i in range(len(in_train))]
    test_data = [(in_test[i], out_test[i]) for i in range(len(in_test))]
    return train_data, test_data


def normalize(data):
    return preprocessing.normalize(data)


def col(matrix: list, i: int) -> list:
    """Return i-th column of matrix."""
    return [row[i] for row in matrix]


colors = ('b', 'g', 'r', 'c',
          'm', 'y', 'k', 'w')


def plot(data, categories, num=1, n_rows=1, n_cols=1, title=None, show=False):
    """Plot 2-dimensional data"""
    plt.subplot(n_rows, n_cols, num)

    i = 0
    for sub in subsets(data, categories):
        plt.plot(col(sub, 0), col(sub, 1), f'{colors[i]}o')
        i += 1

    if title:
        plt.title(title)

    if show:
        plt.show()


def subsets(data, categories):
    for c in categories:
        yield col(subset(data, c), 0)


def subset(data, category):
    return [row for row in data if list(row[-1]) == category]


def line_eq(w, c):
    a = -w[0] / w[1]
    b = -c / w[1]
    return a, b
