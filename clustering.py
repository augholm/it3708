import sklearn.neighbors
import sklearn.cluster
import numpy as np  # noqa


def nearest_depot(depots, customers):
    '''
    returns an np.array of integers.
    '''
    D, C = depots, customers

    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(D[:,[1,2]], [0,1,2,3])
    Y = clf.predict(C[:,[1,2]])
    return Y
