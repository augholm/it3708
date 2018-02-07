import sklearn.neighbors
import sklearn.cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import termcolor


def profile_value(profile, step, num_generations):
    if step == num_generations:
        return profile[-1]
    if not is_iterable(profile):
        return profile
    T = num_generations // len(profile)
    T = step / num_generations * len(profile)

    return profile[int(T)]


def get_population_size(conf_entry, time_step=0):
    conf = conf_entry

    if is_iterable(conf['population_profile']):
        T = conf['num_generations'] // len(conf['population_profile'])
        T = (time_step / conf['num_generations']) * len(conf['population_profile'])
        current_population_size = conf['population_profile'][int(T)]
    else:
        current_population_size = conf['population_profile']
    return current_population_size


def cprint(s):
    '''
    Example usage:
    cprint('[y]Hello [g]world[r]!!!')
    cprint('this text is in [r]red, and this is [y]yellow :)')
    '''
    C = {'r': 'red',
         'g': 'green',
         'b': 'blue',
         'y': 'yellow',
         'm': 'magenta',
         'c': 'cyan',
         'w': 'white'}
    L = re.split('(\[\w\])', s)
    color = 'white'

    s_colored = []
    for each in L:
        if re.match('\[\w\]', each):
            color = C[each[1]]
        else:
            s_colored.append(termcolor.colored(each, color))

    print(''.join(s_colored))


class NoPathFoundException(Exception):
    pass




def flatten(iterable):
    flat_list = [item for sublist in iterable for item in sublist]
    return flat_list


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


# TODO: maybe fix this. Doesnt look good.
def get_by_ids(data, id):
    '''
    data: np.array of shape (N, d) where d is the number of attributes.
    The id must reside in the 0th column.
    id: array-like or integer, specifying which entry to get

    Returns the row that has id equal to `id`. If it does not exist, then it will
    probably return some weird error

    Example usage:
        some_depots = get_by_ids(depots_and_customers, [51, 52, 53])

    returns a np.array
    '''

    id = np.array(id, dtype=np.int64)
    return data[id-1, :]
    # if len(id) == 1:
    #     return get_by_ids(data, id[0])

    # X = pd.DataFrame(data[:, 1:], index=data[:, 0]).loc[id]
    # return np.vstack([X.index.values, X.values.T]).T


def visualize_closest_depot(customers, depots, ax=None):
    # TODO: remove soon
    if ax is not None:
        plt.sca(ax)
    plt.clf()
    D, C = depots, customers
    n_depots = D.shape[0]

    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(D[:,[1,2]], np.arange(n_depots))
    Y = clf.predict(C[:,[1,2]])
    plt.scatter(C[:,1], C[:,2], c=Y)
    plt.scatter(D[:,1], D[:,2], s=150, c='m', marker='*')

    xdims = min(C[:, 1]), max(C[:,1])
    ydims = min(C[:,2]), max(C[:,2])
    xx, yy = np.meshgrid(np.linspace(xdims[0], xdims[1], 100),
                         np.linspace(ydims[0], ydims[1], 100))
    feed = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = clf.predict(feed)
    plt.scatter(xx.ravel(), yy.ravel(), c=Z, alpha=0.1)


def choose_random_paths(individual, n=1, min_length=1, depot_id=None, include_indices=False, include_depot=False):
    '''
    From an individual, choose random `n` random paths each with length at
    least `min_length`.

    depot_id can be one of:
        int           if a signle depot is considered
        array-like    if multiple depots are considered
        'random'      if a random depot is considered
        None or 'all' if all depots are considered

    include_indices, Boolean. If True then will return an array consisting
    of tuples (path, d_idx, p_idx)

    include_depot, Boolean. If True then each path will be prepended
    and appended with the depot index.

    raises NoPathFoundException if no path filling the requirements
    are found.

    returns a list of paths or a list of tuples (d_idx, p_idx, path)
    '''

    if depot_id == 'random':
        g = [np.random.choice(individual.depots)]
    elif type(depot_id) != str and is_iterable(depot_id):
        g = depot_id
    elif depot_id is None or depot_id == 'all':
        g = individual.depots
    elif not is_iterable(depot_id):  # must be an iterable
        g = [depot_id]

    X = []
    for depot_id in g:
        for p, c, cap, d_idx, p_idx in individual.iter_paths(depot_id=depot_id, include_depot=include_depot, yield_depot_idx=True, yield_path_idx=True):
            if len(p) < min_length:
                continue

            if include_indices:
                yld = p, d_idx, p_idx
            else:
                yld = p

            X.append(yld)

    if len(X) < n:  # TODO: consider raising if zero has been found instead of `n`?
        raise NoPathFoundException()

    indices = np.random.choice(range(len(X)), size=n, replace=False)
    return [X[idx] for idx in indices]


def choose_random_depot(individual, n_depots=1, min_num_paths=1):
    '''
    From an individual, randomly chooses a depot that has at least `min_num_paths` in it.
    This is the same as requiring that you have at least `min_num_paths` trucks on that depot.

    returns an int if n_depots=1, otherwise an np.array of ints of size `n_depots`

    '''
    candidates = []
    for depot_id in individual.iter_depots():
        if len([_ for _ in individual.iter_paths(depot_id)]) >= min_num_paths:
            candidates.append(depot_id)

    if n_depots == 1:
        return np.random.choice(candidates)
    else:
        return np.random.choice(candidates, size=n_depots)


def euclidean_dist(X):
    '''
    X: np.array of shape (N, d) where d is the dimensionality (e.g in 2D-case d=2)
    and N is the number of points.

    Calculates the total distance if straight lines were drawn between each
    subsequent point.

    returns a float

    '''
    if X.shape[0] <= 1:
        raise Exception('Cannot find distance with less than 2 points of data')

    dists = np.sqrt((np.diff(X, n=1, axis=0)**2).sum(axis=1))
    return dists.sum()


def all_euclidean_dist(X):
    '''
    excpeted: Nx2 array of xy-coordinates.
    '''
    L = []
    for i, each in enumerate(X):
        relative_locs = X[:] - each
        distances = np.sqrt(np.sum(relative_locs ** 2, axis=1))
        L.append(distances)
    return np.matrix(L)


def delete_by_value(arr, value):
    '''
    Returns an array where all instances of `value` in `arr` are gone
    '''
    arr = np.array(arr)
    return np.delete(arr, np.argwhere(arr == value).squeeze())


def matrix_row_to_array(row):
    return np.asarray(row).squeeze()
