import pathlib
import re


def parse_config(filename, verbose=False):
    if filename.__class__ == pathlib.PosixPath:
        filename = str(filename)

    with open(filename) as f:
        L = list(f)

    D = {}
    for line in L:
        if re.match('\s*#', line):  # drop lines that contain comment
            continue  # ignore comments

        if ':' not in line:  # line is useless, skip
            continue

        key, value = line.strip().strip(' ').split(':')
        if value.endswith(','):
            value = value[:-1]
        value = value.strip(' ')

        is_array = re.match('\[.*\]', value) is not None
        if is_array:
            the_array = eval(re.match('\[.*\]', value).group())
            D[key] = the_array
            continue

        is_numeric = re.match('^\d+\.?\d*$', value) is not None
        if is_numeric:
            if '.' in value:
                D[key] = float(value)
            else:
                D[key] = int(value)
            continue

        is_boolean = re.match('(true|false)', value.lower()) is not None
        if is_boolean:
            if value.lower() == 'true':
                D[key] = True
            else:
                D[key] = False
            continue

        # else it's a string
        D[key] = value

    if verbose:
        print_config(D)
    return D


def print_config(config):
    print('---------- PARAMS ------------')
    for k, v in config.items():
        print(f'{k}: {v}')
        #  print(f'{k}: {v}')
    print('------------------------------')
