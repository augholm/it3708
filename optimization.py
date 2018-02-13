import utils
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from tqdm import tqdm
import utils

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
        return np.hstack((np.arange(X.shape[0]), 0))
        # return np.arange(X.shape[0])

    try:
        hull = ConvexHull(X)
    except:
        utils.cprint('[y]Warning: cannot solve TSP; too few points. Aborting')
        return np.concatenate((np.arange(X.shape[0]), [0]))
        

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
    # import ipdb; ipdb.set_trace()
    # vertices = utils.delete_by_value(vertices, n_points-1)

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
            vertices = np.concatenate((vertices[idx:-1], vertices[:idx+1]))
            pass

    if circular_indices:
        return np.hstack((vertices, vertices[0]))
    else:
        return vertices


def split(X, Q, capacity, input_args_check=True):
    '''
    X np.array of shape (t+2). Consists of the xy-coordinates of the t customers,
    as well as the depot which is located in X[0] and X[-1].

    Q: np.array of shape t, consisting of the capacities
    capacity: int, the upper limit of the capacity

    Example:
        X = array([[0, 0, 100],
                   [0, 5, 3],
                   [7, 7, 50]])
        depot is at (0, 0) with supply 100. Customer 0 is at (0, 5) with demand 3, etc.

    returns: (L, cost)
        L: list of np.arrays, consists of the indices for each subtour
        cost: total cost (euclidean)
    '''

    '''
    Step 0 --- check if data is correct
    '''
    if input_args_check:
        if not Q.min() >= 0:
            raise ValueError('Q must be non-negative')

        # if not np.all(X[0, :] == X[-1, :]):
        #     raise ValueError('''Expected first and last row of X to be equal
        #     as they represent the coordinate of the depot.''')

        if capacity < Q.max():
            raise ValueError('''Cannot be solved as the highest demand from a
            customer exceeds capacity''')

    '''
    Step 1 --- sort according to a big tour and find distance matrix
    '''
    indices = solve_tsp(X, start_index=0, plot=False)
    indices = indices[:-1]

    # knowing that `indices` also includes the depot at the beginning and end,
    # we need to remove the depot (index 0) and make index 1 to index 0 by
    # subtracting.
    _maxint = np.iinfo(np.int64).max

    customer_indices = indices[1:-1] - 1

    S = indices
    Q = np.hstack((_maxint, Q))
    # Q = Q[customer_indices]
    D = utils.all_euclidean_dist(X[indices])
    V = np.ones(X.shape[0], np.int64) * _maxint
    V[0] = 0
    P = np.ones(X.shape[0], np.int64) * -1

    I = np.random.permutation(indices)

    '''
    Step 2 --- solve this boy
    '''

    # we can be certain to not have a path that exceeds b units
    b = int(capacity / np.min(Q))  # noqa
    
    # import ipdb; ipdb.set_trace()
    P[S[1]] = 0  # set this guy to the depot
    for i_idx, i in enumerate(S[1:], 1):
        load = Q[i]
        cost = D[0, i] + D[i, 0]
        if V[i] == _maxint:
            V[i] = cost
        for j_idx, j in enumerate(S[i_idx+1:], i_idx+1):
            load += Q[j]
            cost = cost - D[S[j_idx-1],0] + D[S[j_idx-1],j] + D[j,0]

            if load <= capacity and V[i] + cost < V[j]:
                V[j] = V[i] + cost
                P[j] = i
            else:
                break  # no point in going further
        if P[i] == -1:
            import ipdb; ipdb.set_trace()
            pass

    L = []

    from_ = S.shape[0]

    import time
    start = time.time()
    # desired: [2,3,4,5], [7,1], [0,6]
    while True:
        if time.time() - start >= 5:
            import ipdb; ipdb.set_trace()
        c = S[from_ - 1]
        parent = P[c]
        if parent == -1:
            break

        from_ = np.asscalar(np.argwhere(S == parent))
        to = np.asscalar(np.argwhere(S == c))
        if parent == 0:
            seq = S[from_+1:to+1]
        else:
            seq = S[from_:to+1]
        L.append(seq)

        if parent == 0:
            break
    cost = V[S[-1]]

    return L, cost



# X = make_problem(25)
# Q = np.random.choice(np.arange(1, 21), size=X.shape[0]-1)
# L, cost = split(X, Q, capacity=100)
# 
# from cycler import cycler
# ax = plt.gcf().get_axes()[0]
# ax.clear()
# ax.set_prop_cycle(cycler('linestyle', '- -- : -.'.split()))
# ax.scatter(X[:, 0], X[:, 1], c='y')
# ax.scatter(X[0, 0], X[0, 1], c='r', marker='*')
# for each in L:
#     each = np.hstack((0, each, 0))
#     ax.plot(X[each, 0], X[each, 1], 'b', alpha=0.7)
# 
