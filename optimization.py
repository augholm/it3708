import utils
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from tqdm import tqdm

plt.ion()


def make_problem(n):
    '''
    Returns a simple Nx2 array of random xy-coordinates. Used for debugging
    '''
    return np.random.randint(1, 100, size=(n, 2))


def solve_tsp(X, start_index=None, circular_indices=False, run_2_opt=False, plot=False, verbose=False):
    '''
    X: np.array of shape (N, 2) where each row corresponds to a xy-coordinate.
    start_index: Boolean, if True then start with `start_index` in the return value
    run_2_opt: Boolean, run additional tweaking to tsp. Takes some more time

    plot: Boolean, if True then shows a plot of the TSP
    verbose: Boolean, if True then adds a tqdm progress bar. Nice for large problems

    returns: np.array of shape (N,) with unique elements, each being an index to
    one of the rows.

    Example:

     X = array([[85,  7],
               [19, 69],
               [99, 50],
               [ 4,  2],
               [66,  6],
               [42, 86],
               [70, 82],
               [76, 67],
               [57,  7],
               [55, 39]])

    returns array([8, 4, 1, 2, 3, 6, 7, 9, 8]) (for example)
    '''

    '''
    pt. 1: Insertion heuristic (convex hull)
    '''
    if X.shape[0] <= 3:
        # utils.cprint('[y]Warning: cannot solve TSP; too few points. Aborting')
        return np.arange(X.shape[0])

    try:
        hull = ConvexHull(X)
    except:
        utils.cprint('[y]Warning: cannot solve TSP; too few points. Aborting')
        return np.arange(X.shape[0])
        

    D = utils.all_euclidean_dist(X)

    all_vertices = np.arange(X.shape[0])
    unvisited = np.setdiff1d(all_vertices, hull.vertices)
    vertices = np.hstack((hull.vertices, hull.vertices[0]))

    def cheapest_insertion(i, seq):
        '''
        returns (cost, j, k)
        where cost, int -- the cost of the insertion (positive is bad)
              j, int -- the index of the element that will go to j
              k, int -- the index of the element that i will go to

        (so we end up with something like this: ... -> j -> i -> k -> ...
        '''
        L = []
        for j, k in zip(seq, seq[1:]):
            old_edge = utils.euclidean_dist(
                X[[j, k]])
            new_edge = D[j, i] + D[i, k]
            cost = -old_edge + new_edge
            L.append((cost, j, k))
        return min(L, key=lambda x: x[0])

    if verbose:
        pbar = tqdm(total=len(unvisited))
    while len(unvisited) > 0:
        if verbose:
            pbar.update(1)
        data = []
        for i in unvisited:
            distances = D[i, np.setdiff1d(all_vertices, unvisited)]
            min_distance = distances.min()
            idx = D[i, :] == min_distance

            cost, j, k = cheapest_insertion(i, vertices)
            data.append((cost, j, i, k))

        cost, j, i, k = min(data, key=lambda x: x[0])

        idx = np.argwhere(vertices == j)[0][0]
        vertices = np.hstack((
            vertices[:idx+1],
            i,
            vertices[idx+1:]))

        unvisited = utils.delete_by_value(unvisited, i)

    if verbose:
        pbar.close()

    '''
    pt. 2  -- tour improvement
    '''

    if plot:
        if not plt.fignum_exists(1):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(X[:, 0], X[:, 1], '.')
        else:
            ax1, ax2 = plt.gcf().get_axes()
            ax1.cla()
            ax2.cla()

        ax1.plot(X[vertices, 0], X[vertices, 1], 'r--', alpha=0.3, label='pt1')
        ax1.plot(X[vertices[0], 0], X[vertices[0], 1], 'ro')
        ax1.set_title('at step 1')

    if run_2_opt:
        def two_opt(i, k, seq):
            new_seq = seq[:i+1], seq[k:i:-1], seq[k+1:]
            return np.hstack(new_seq)

        best_cost = D[vertices[:-1], vertices[1:]].sum()
        for i, j in itertools.combinations(np.arange(1, len(vertices)-1), 2):
            # iterate for every combinations, except start and ending point

            new_path = two_opt(i, k, np.array(vertices, copy=True))
            cost = D[new_path[:-1], new_path[1:]].sum()
            if cost < best_cost:
                if verbose:
                    print('hey found shorter')
                vertices = np.array(new_path, copy=True)
                best_cost = cost

    if plot:
        ax1.plot(X[:, 0], X[:, 1], '.')
        ax1.plot(X[vertices, 0], X[vertices, 1], 'g--', alpha=0.3, label='pt2')
        ax1.plot(X[vertices[0], 0], X[vertices[0], 1], 'ro')

    # get rid of the 'end' of the path, because we know it's the same as the first
    n_points = X.shape[0]
    vertices = utils.delete_by_value(vertices, n_points-1)

    if start_index is not None:
        if start_index == vertices[0]:
            pass
        else:
            '''
            example:
                vertices =     [6, 7, 0, 5, 2, 4, 8, 3, 9, 1, 6]
                start_index = 3
                --> vertices = [3, 9, 1, 6, 7, 0, 5, 2, 4, 8, 3]
            '''
            idx = np.argwhere(vertices == start_index)[0][0]
            vertices = np.hstack((vertices[idx:], vertices[1:idx], vertices[idx]))

    if circular_indices:
        return np.hstack((vertices, vertices[0]))
    else:
        return vertices



X = make_problem(8)
sol2 = solve_tsp(X, start_index=0, plot=True, verbose=True, run_2_opt=True)
