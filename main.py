from data import loader
import nn
from util import *

categories = [[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]]


def main():
    male, female = loader.read_bmi('bmi_data.txt')

    dataset = male
    train_data, test_data = split(dataset, test_size=0.4)

    plot(dataset, categories, title='Data', show=True)

    n_in = len(dataset[0][0])
    n_out = len(dataset[0][1])
    model = nn.Model(
        nn.Linear(n_in, 8, nn.sigmoid),
        nn.Linear(8, 8, nn.sigmoid),
        nn.Linear(8, n_out, nn.sigmoid)
    )

    bmi = nn.Classifier(model)
    bmi.train(train_data, test_data, 0.93)

    res = result(bmi, dataset)

    plot(dataset, categories, n_cols=2, title='Data')
    plot(res, categories, num=2, n_cols=2, title='Prediction', show=True)


def result(classifier, data):
    normal_in = normalize(col(data, 0))

    res = []
    i = 0
    for x in normal_in:
        raw = data[i][0]
        out = classifier(x)
        res.append((raw, out))
        i += 1
    return res


if __name__ == '__main__':
    main()
