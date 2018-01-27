import sklearn.neighbors
import sklearn.cluster
import numpy as np  # noqa


def nearest_depot(depots, customers):
    '''
    Finds the depot id that is closest to the customers.

    returns an np.array of integers, where each integer represent
    the index of the depot that is closest for the customer
    '''

    depot_ids = depots[:, 0]
    depot_coords = depots[:, [1, 2]]

    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(depot_coords, depot_ids)

    customer_coords = customers[:, [1, 2]]
    Y = clf.predict(customer_coords)
    return Y


