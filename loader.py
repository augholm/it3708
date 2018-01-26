import re
import numpy as np


def load_dataset(filename):
    '''
    Loads a dataset that exists in `filename` and returns the necessary
    information from it.

    filename:
        path to filename, e.g 'data/problem/p01'

    returns the tuple (depots, customers)
        customers: np.array of shape (N, 5)
            where N is the number of customers
            where each row is (i, x, y, d, q)
                i: int, customer number
                x: int, x-coordinate
                y: int, y-coordinate
                d: int, necessary service duration required for this customer
                q: int, demand for this customer
            
        depots: np.array of shape (N, 8)
            where N is the number of depots
            where each row is (i, x, y, d, q, m, D, Q)
                i, x, y, d, q as usual.
                m: int, maximum number of vehicles available for the depot
                D: int, maximum duration of a route (0 if unrestricted)
                Q: int, maximum load for a vehicle in that depot (typically 80)
    '''
    L = []
    with open(filename) as f:
        for line in f:
            parsed_line = list(map(int, re.split('\s+', line.strip())))
            if len(parsed_line) >= 5:
                parsed_line = parsed_line[:5]
            L.append(parsed_line)

    m, n, t = L[0]
    X = np.array(L[1:1+t])
    D, Q = X[:, 0], X[:, 1]
    X = np.array(L[t+t:])

    customers = X[:-t]

    m = np.ones((t, 1)) * m
    D = D.reshape((t, 1))
    Q = Q.reshape((t, 1))
    depots = np.hstack([X[-t:, :], m, D, Q])

    return (depots, customers)
