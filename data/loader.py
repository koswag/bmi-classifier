from random import random

_path = r'C:\Users\kokos\Desktop\bmi\data'


def set_path(path):
    global _path
    _path = path


def read_bmi(fname):
    male, female = [], []
    path = _path + f'\\{fname}'
    classes = ('niedowaga', 'norma', 'nadwaga')

    with open(path) as file:
        for line in file:
            values = line.split(',')

            gender = values[0]
            cl = name_to_value(values[-1], classes)
            values = [float(val) for val in values[1:-1]]

            entry = values, cl
            if cl != [1, 0, 0] or (entry[0][1] < 110 and random() > 0.5):
                if gender == "Male":
                    male.append(entry)
                elif gender == "Female":
                    female.append(entry)

    return male, female


def read(fname, classes=None):
    data = []
    path = _path + f'\\{fname}'
    with open(path) as file:
        for line in file:
            values = line.split(',')

            cl = name_to_value(values[-1], classes)
            values = [float(val) for val in values[:-1]]

            entry = list(values), cl
            data.append(entry)
    return data


def name_to_value(name, classes):
    name = name.strip()
    if classes:
        if name == f'{classes[0]}':
            return [0, 0, 1]
        elif name == f'{classes[1]}':
            return [0, 1, 0]
        elif name == f'{classes[2]}':
            return [1, 0, 0]
    else:
        return name
