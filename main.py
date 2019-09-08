from data import loader
import nn
from util import *

categories = [[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]]

file = r'data\bmi_data.txt'


def main():
    path = get_path(file)
    male, female = loader.read_bmi(path)

    dataset = male
    plot(dataset, title='Data', show=True)
    n_in = len(dataset.inputs[0])
    n_out = len(dataset.outputs[0])

    train_data, test_data = split(dataset, test_size=0.4)

    model = nn.Model(
        nn.Linear(n_in, 32, nn.sigmoid),
        nn.Linear(32, 8, nn.sigmoid),
        nn.Linear(8, n_out, nn.sigmoid)
    )

    bmi = nn.Classifier(model)
    bmi.train(train_data, test_data, target_acc=0.92)

    res = result(bmi, dataset)

    plot(dataset, categories, n_cols=2, title='Data')
    plot(res, categories, title='Prediction', show=True, num=2, n_cols=2)


def result(classifier: nn.Classifier, data: DataSet):
    normal_in = normalize(data.inputs)

    res = DataSet.empty()
    i = 0
    for x in normal_in:
        raw = data.inputs[i]
        out = classifier(x)
        res.add(raw, out)
        i += 1
    return res


if __name__ == '__main__':
    main()
