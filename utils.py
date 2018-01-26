import sklearn.neighbors
import sklearn.cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        my_customer = get_by_id(customers, 22)
        some_depots = get_by_ids(depots, [51, 52, 53])

    returns a np.array
    '''
    if not is_iterable(id):
        match = data[data[:, 0] == id].squeeze()
        return match

    if len(id) == 1:
        return get_by_ids(data, id[0])

    X = pd.DataFrame(data[:, 1:], index=data[:, 0]).loc[id]
    return np.vstack([X.index.values, X.values.T]).T


def visualize_closest_depot(customers, depots, ax=None):
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


def visualize_chromosome(chromosome, customers, depots, ax=None):
    if ax is not None:
        plt.sca(ax)
    plt.clf()
    visualize_closest_depot(customers, depots)

    D, C = depots, customers
    locs = np.vstack([
        customers[:, [0, 1, 2]],
        depots[:, [0, 1, 2]]
        ])
    cmap = 'r g b y m'.split()
    for tour, dept_id in zip(chromosome, depots[:, 0]):
        tour = np.array(tour)
        for subtour, c in zip(np.split(tour, np.where(tour == 0)[0]),cmap):
            if len(subtour) == 0:
                continue
            subtour[0] = dept_id
            subtour = np.array(subtour.tolist() + [dept_id])
            X = get_by_ids(locs, subtour)[:, [1, 2]]
            print(c)
            plt.plot(X[:,0], X[:,1], c, alpha=0.3)



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
