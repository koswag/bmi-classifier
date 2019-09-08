from util import *

_bmi_classes = ('niedowaga', 'norma', 'nadwaga')


def read_bmi(path):
    male, female = DataSet.empty(), DataSet.empty()
    with open(path) as file:
        for line in file:
            gender, values, cls = parse_bmi(line)

            if gender == "Male":
                male.add(values, cls)
            elif gender == "Female":
                female.add(values, cls)
            else:
                raise ValueError(f'{gender} is not a gender.')

    return male, female


def parse_bmi(line):
    values = line.split(',')

    gender = values[0].strip()
    values = [float(val) for val in values[1:-1]]
    cls = name_to_value(values[-1], _bmi_classes)

    return gender, values, cls


def read(path, classes=None):
    entries = DataSet.empty()
    with open(path) as file:
        for line in file:
            values, cls = parse_line(line, classes)
            entries.add(values, cls)
    return DataSet(entries)


def parse_line(line, classes):
    values = line.split(',')

    values = [float(val) for val in values[:-1]]
    cls = name_to_value(values[-1], classes)

    return values, cls


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
