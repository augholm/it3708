import sklearn.neighbors
import sklearn.cluster
import numpy as np  # noqa


def nearest_depot(X, depots):
    '''
    depots: the indices in X for the depots
    '''
    # depot_coords = X[depots][:, [-1, 1]]
    depot_coords = X[depots, :2]
    clf = sklearn.neighbors.NearestCentroid()
    clf.fit(depot_coords, depots)

    customers_mask = np.ones_like(X[:, -1], np.bool)
    customers_mask[depots] = False
    customer_locs = X[customers_mask, 0:2]

    #Add noise to customer locations
    for customer in customer_locs:
    	for coord in customer:
    		coord += np.random.normal(loc=coord)

    Y = clf.predict(customer_locs)
    return Y
