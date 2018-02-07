import itertools
import collections
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

import optimization
import clustering
import utils


class Individual():
    def __init__(self, X, depots, D, initialize=True):
        '''
        X: np array of shape (N+M, 5) -- each row is (x, y, z, a, b) and there are 
            x: x-coordinate
            y: y-coordinate
            z: demand if customer, supply if depot
            a: the depot it belongs to
            b: path index for that depot

        depots: np.array of indices in X that is the depot. shape (M,)

        D: np.array of shape (M+N, M+N)

        '''

        self.X = X
        self.D = D
        self.depots = depots

        '''
        paths: dictionary of arrays. Each element in array is np.array
        consisting of indices in the path

        '''
        self.paths = collections.defaultdict(lambda: [])
        self.costs = {}
        self.caps = collections.defaultdict(lambda: [])

        if initialize:
            self.generate_initial_state()

    def generate_initial_state(self):
        Y = clustering.nearest_depot(self.X, self.depots)
        for d_idx in self.depots:
            supply = self.X[d_idx, 2]
            L = []
            indices = np.argwhere(Y == d_idx).squeeze()
            np.random.shuffle(indices)
            partition = np.cumsum(self.X[indices, 2]) // supply
            for each in np.unique(partition):
                path = indices[partition == each]
                path = np.hstack((d_idx, path, d_idx))
                # path = self.solve_tsp(path)
                L.append(path[1:-1])

            self.paths[d_idx] = L

        for p, _, _, d_idx, p_idx in self.iter_paths(include_depot=False,
                                                     yield_depot_idx=True,
                                                     yield_path_idx=True):
            self.update_path(p, d_idx, p_idx)

    ##############################################################################
    # utility functions
    ##############################################################################

    def update_path(self, path, d_idx, p_idx):
        '''

        '''
        if len(path) > 0:
            if path[0] == path[-1] and len(path) > 1:
                pass
            else:
                path = np.hstack((d_idx, path, d_idx))

        cost = self.cost(path)
        cap = self.capacity(path)

        self.paths[d_idx][p_idx] = path[1:-1]
        self.costs[(d_idx, p_idx)] = cost
        self.caps[(d_idx, p_idx)] = cap

    def capacity(self, path):
        if len(path) == 0:
            return 0

        if path[0] != path[-1]:
            raise Exception('must include depot')
        return self.X[path[1:-1], 2].sum()

    def cost(self, path):
        if len(path) == 0:
            return 0

        if path[0] != path[-1]:
            raise Exception('must include depot')

        path = np.array(path)

        return self.D[path[:-1], path[1:]].sum()

    def fitness_score(self):
        return sum(list(self.costs.values()))

    def find_customer(self, i):
        '''
        returns the depot index and path index of our customer
        '''
        for x in self.iter_paths(yield_depot_idx=True, yield_path_idx=True):
            p, c, cap, d_idx, p_idx = x
            if i in p:
                return d_idx, p_idx

    def iter_paths(self, depot_id=None, include_depot=False, yield_depot_idx=False, yield_path_idx=False):
        '''
        yields (path, cost, capacity, [depot_idx], [path_idx])
            where path is the sequence of customers visited
            cost is the cost for that specific path
            capacity is the capacity required for the path
            yield_depot_idx is the index of the depot
            yield_path_idx: is the index of the path
        '''
        if depot_id is not None:
            if len(self.costs) > 0:
                keys = filter(lambda x: depot_id in x, self.costs.keys())
                keys = list(keys)
                if len(keys) == 0:
                    a = []
                else:
                    a = [self.costs[key] for key in keys]
            else:
                a = []

            if len(self.caps) > 0:
                keys = filter(lambda x: depot_id in x, self.caps.keys())
                keys = list(keys)
                if len(keys) == 0:
                    b = []
                else:
                    b = [self.caps[key] for key in keys]
            else:
                b = []

            n_paths = len(self.paths[depot_id])
            iterable = itertools.zip_longest(
                    self.paths[depot_id], a, b,
                    fillvalue=None)

            for i, (p, c, cap) in enumerate(iterable):
                if include_depot:
                    p = np.hstack((depot_id, p, depot_id))
                else:
                    if len(p) > 0 and p[0] == p[-1] and len(p) > 1:
                        pass
                yld = [p, c, cap]
                if yield_depot_idx:
                    yld.append(depot_id)
                if yield_path_idx:
                    yld.append(i)

                yield yld
        else:
            for d_idx in self.paths.keys():
                for each in self.iter_paths(depot_id=d_idx, include_depot=include_depot, yield_depot_idx=yield_depot_idx, yield_path_idx=yield_path_idx):
                    yield each

    def move_customer(self, i, intra_depot=False, strategy='best'):
        '''
        i: the index of the customer
        strategy: 'best'|'random'
        '''

        d_idx, p_idx = self.find_customer(i)
        path_without_customer = utils.delete_by_value(self.paths[d_idx][p_idx], i)
        self.update_path(path_without_customer, d_idx, p_idx)

        demand = self.X[i, 2]
        supply = self.X[d_idx, 2]

        # TODO: maybe allow infeasible insertions too
        if intra_depot:
            L = []
            for p, c, cap, d_id, p_id in self.iter_paths(depot_id=d_idx, yield_depot_idx=True, yield_path_idx=True):
                if cap + demand > supply:
                    continue

                path_with_customer = np.hstack((d_id, p, i, d_id))
                path_candidate = self.solve_tsp(path_with_customer)
                cost = self.cost(path_candidate)
                L.append((cost, path_candidate, d_id, p_id))
        else:
            L = []
            for p, c, cap, d_id, p_id in self.iter_paths(depot_id=None, yield_depot_idx=True, yield_path_idx=True):
                if cap + demand > supply:
                    continue

                path_with_customer = np.hstack((d_id, p, i, d_id))
                path_candidate = self.solve_tsp(path_with_customer)
                cost = self.cost(path_candidate)
                L.append((cost, path_candidate, d_id, p_id))

        if len(L) == 0:
            raise Exception('hey this seems wrong')

        L = sorted(L, key=lambda x: x[0])
        cost, path, d_id, p_id = L[0]
        if d_id == d_idx and p_id == p_idx and len(L) > 1:
            cost, path, d_id, p_id = L[1]
        self.update_path(path, d_id, p_id)

        return

    def solve_tsp(self, path):
        if path[0] != path[-1] or len(path) <= 1:
            raise Exception('wrong type path')

        indices = optimization.solve_tsp(self.X[path, 0:2], circular_indices=False, start_index=0)
        new_path = path[indices]
        if len(np.unique(new_path[1:-1])) != len(new_path[1:-1]):
            import ipdb; ipdb.set_trace()
        if path[0] != path[-1] and len(path) > 1:
            import pdb; pdb.set_trace() # shouldnt occur

        return new_path

    def visualize(self, ax=None, title=None):
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_prop_cycle(cycler('linestyle', '- -- : -.'.split()))

        depots = self.X[self.depots]
        mask = np.ones_like(self.X[:, 0])
        mask[self.depots] = 0
        customers = self.X[mask]

        ax.scatter(depots[:, 0], depots[:, 1], c='r', marker='*')
        ax.scatter(customers[:, 0], customers[:, 1], s=2*customers[:, 2], c='b', marker='.')
        # for txt, x, y in zip(self.customers[:, 0], self.customers[:, 1], self.customers[:, 2]):
        #     txt = str(txt)
        #     ax.annotate(txt, (x, y), size=8)

        for color, depot in zip('c m y k r g b'.split(), depots):
            for path, _, _ in self.iter_paths(depot_id=None, include_depot=True):
                xy_coords = self.X[path][:, 0:2]
                ax.plot(xy_coords[:, 0], xy_coords[:, 1], color, alpha=0.6)
        if title is not None:
            ax.set_title(title)

