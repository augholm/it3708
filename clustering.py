import sklearn.neighbors
import sklearn.cluster
import numpy as np  # noqa


def nearest_depot(X, depots):
    depot_coords = X[depots][:, [-1, 1]]
    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(depot_coords, depots)

    customers_mask = np.ones_like(X[:, -1], np.bool)
    customers_mask[depots] = False
    customer_locs = X[customers_mask, 0:2]

    Y = clf.predict(customer_locs)
    return Y
