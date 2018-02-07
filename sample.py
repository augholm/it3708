import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import utils
import collections
import clustering
L = []

plt.ion()


class Individual():
    def __init__(self, customers, depots, initialize=True, path_cost_dict={}, distance_matrix=None):
        '''
        customers: np.array representing the customers
        depots: np.array representing the depots
        initialize: boolean, if True will call self.generate_initial_state
        path_cost_dict: used to optimize path_cost
        distance_matrix: lookup for further optimize path cost.
        '''
        self.distance_matrix = distance_matrix
        self.customers = customers
        self.depots = depots
        self.tours = collections.defaultdict(lambda: [])
        self.path_cost_dict = path_cost_dict
        if initialize:
            self.generate_initial_state()

    def generate_initial_state(self):
        '''
        Populates self.tours.

        Description of algorithm
        (1) Based on the locations of the depots and customers, find the closest
            depot for each customer

        (2) For each depot (and the customers that are assigned to it), shuffle the
            order of the customers and make a series of paths, that each do not
            exceed the allowed carry limit for the trucks.

        note: No change in tours are done after step 2 in this algorithm.

        returns: nothing, but populates self.tours
        '''
        Y = clustering.nearest_depot(self.depots, self.customers)

        def split_to_paths(sequence, depot_id, carry_limit):
            '''
            sequence: list of customer indices, without depot. Example: [5, 32, 22]
            depot_id: integer, representing the depot id
            carry_limit: the maxmimum allowed carry for a particular route

            returns: a list of of subsequences, each with total capacity below
            maximum allowed

            Description of algorithm: While there still are customers that have
            not been served, add them to path. THe selection of the next
            customer from the `current` customer is based on the shortest
            distance from `current`. If the route can also serve the next
            customer (i.e the carry limit has not been exceeded), then the next
            customer is also served. Otherwise return the route and start a new
            route.
            '''
            Q = []
            current = depot_id
            unvisited = sequence
            path = [depot_id]
            while len(unvisited) > 0:
                while self.capacity_requirement(path + [depot_id]) < carry_limit and len(unvisited) > 0:  # noqa
                    distances = self.distance_matrix[np.int64(current)-1, :]
                    idx = np.argmin(distances[0, unvisited-1])
                    current = unvisited[idx]
                    path.append(current)
                    unvisited = np.delete(unvisited, idx)
                if len(unvisited) == 0:
                    if self.capacity_requirement(path + [depot_id]) > carry_limit:
                        Q.append(np.array(path[1:-1], np.int64))
                        Q.append(np.array(path[-1], np.int64))
                        return Q
                    else:
                        Q.append(np.array(path[1:], np.int64))
                else:
                    Q.append(np.array(path[1:-1], np.int64))
                path = [depot_id, current]
            return Q

        for dept_id, carry_limit in self.depots[:, [0, 7]]:
            assignment = self.customers[Y == dept_id][:, 0]
            A = np.random.permutation(assignment)
            X = split_to_paths(A, dept_id, carry_limit)
            self.tours[np.int64(dept_id)] = X

    def iter_paths(self, depot_id, include_indices=False, include_depot=False):
        '''
        include_indices, boolean. If True then will also yield the depot_id and
        index of the specific path as a tuple. Otherwise only returns the path.

        include_depot: boolean, if true then also include the
        depot id in the beginning / end

        Usage:
            I = Individual(customers, depots)
            for tour in I.iter_paths(52, include_depot=False):
                >> [42, 65, 24]
                >> [38, 66, 12]
                >> ...

            for tour in I.iter_paths(52, include_depot=True):
                >> [52, 42, 65, 24, 52]
                >> [52, 38, 66, 12, 52]
                >> ...

        '''
        for path_id, tour in enumerate(self.tours[depot_id]):
            if not utils.is_iterable(tour):
                tour = [tour]
            if include_depot:
                tour = np.hstack([np.array([depot_id]), np.array(tour), np.array([depot_id])])
            else:
                tour = np.array(tour)

            if include_indices:
                yield depot_id, path_id, tour
            else:
                yield tour
        pass

    def iter_depots(self):
        '''
        Yields 51, 52, ...
        '''
        depot_ids = self.tours.keys()
        for each in depot_ids:
            yield each

    def capacity_requirement(self, tour):
        '''
        For a given tour (represented by the customer indices), finds the
        total demand by all customers on that tour. The demand for each
        customer is found by the 5th column in `customers`.

        Example:
            tour = array([1, 5, 8])
            customers = array([[ 1, 37, 52,  0,  7],
                               [ 2, 49, 49,  0, 30],
                               [ 3, 52, 64,  0, 16],
                               [ 4, 20, 26,  0,  9],
                               [ 5, 40, 30,  0, 21],
                               [ 6, 21, 47,  0, 15],
                               [ 7, 17, 63,  0, 19],
                               [ 8, 31, 62,  0, 23],
                               [ 9, 52, 33,  0, 11],
                               [10, 51, 21,  0,  5],
                               [11, 42, 41,  0, 19],
            >> returns 7 + 21 + 23 = 51

        returns a float
        '''
        if len(tour) == 0:
            return 0

        stacked = np.vstack([
            self.customers,
            self.depots[:, :5],
            ])
        capacities = utils.get_by_ids(stacked, tour)
        if len(capacities.shape) == 1:
            return capacities[1]
        else:
            return capacities[:, 4].sum()

    def _find_customer(self, customer_id):
        '''
        Returns the path index and depot index that the given customer is on.
        '''
        for d_idx in self.iter_depots():
            for p_idx, path in enumerate(self.iter_paths(d_idx)):
                if customer_id in path:
                    return (d_idx, p_idx)

    def move_customer(self, customer_id, intra_depot=True, stochastic_choice=False):
        '''
        Moves a single customer (given customer_id) to a different (and
        feasible) path such that the the insertion of the customer on that
        path gives the greatest cost reduction.

        returns: nothing, but self.routes are changed
        '''
        # step 1: find the customer and remove it from the current tour
        d_idx, p_idx = self._find_customer(customer_id)
        _d_idx, _p_idx = d_idx, p_idx  # noqa
        path_with_customer = np.array(self.tours[d_idx][p_idx])
        path_without_customer = utils.delete_by_value(path_with_customer, customer_id)

        self.tours[d_idx][p_idx] = path_without_customer

        # step 2: find somewhere to put this boy
        stacked = np.vstack([
            self.customers,
            self.depots[:, :5],
            ])
        customer = utils.get_by_ids(stacked, customer_id)
        demand = customer[4]
        L = []

        g = [d_idx] if intra_depot else self.iter_depots()
        for d_idx in g:
            carry_lim = self.depots[self.depots[:, 0] == d_idx].squeeze()[7]

            for p_idx, path in enumerate(self.iter_paths(d_idx, include_depot=True)):
                if self.capacity_requirement(path) + demand <= carry_lim:
                    path_candidate = self.optimally_insert(path, customer_id,
                                                           stochastic_choice=stochastic_choice)
                    score = self.path_cost(path_candidate)
                    L.append((score, d_idx, p_idx, path_candidate))

        if stochastic_choice:
            scores = np.array([e[1] for e in L])
            distr = np.abs(np.log(scores / scores.sum())).cumsum()
            v = np.random.uniform(distr.min(), distr.max())
            idx = np.argwhere(distr >= v)[0][0]
            min_score, min_d_idx, min_p_idx, path = L[idx]

        else:
            min_score, min_d_idx, min_p_idx, path = min(L, key=lambda x: x[0])

        if min_d_idx == _d_idx and min_p_idx == _p_idx and len(L) >= 2:
            # make sure it is unique.
            L = sorted(L, key=lambda x: x[0])
            min_score, min_d_idx, min_p_idx, path = L[1]

        path = path[1:-1]  # cut away the beginning / end
        self.tours[min_d_idx][min_p_idx] = path

    def optimally_insert(self, path, customer_id, stochastic_choice=False):
        '''
        path: list of ints (customer id's)
        customer_id: int, the customer we want to place
        stochastic_choice: Boolean. If True, then a suboptimal insertion may
        occur with a small probability

        returns: a list of ints


        Example:
            path = [50, 1, 2, 3, 50]
            customer_id = 25

            considers:
                min([cost([50, 25, 1, 2, 3, 50]),
                     cost([50, 1, 25, 2, 3, 50]),
                     cost([50, 1, 2, 25, 3, 50]),
                     cost([50, 1, 2, 3, 25, 50]))
                >> returns the minimum of these, e.g [50, 1, 25, 2, 3, 50], if
                that has the lowest cost

        ---------------------

        '''
        path = np.array(path)
        L = len(path)

        '''
        Example:
        path = [50, 1, 2, 3, 50]
        customer = 4

        --> path_cands = array([50, 4, 1, 2, 3, 50],
                               [50, 1, 4, 2, 3, 50],
                               [50, 1, 2, 4, 3, 50],
                               [50, 1, 2, 3, 4, 50])
        '''
        indices = np.arange(1, L**2 - 2, L+1)
        X = np.vstack([path] * (L-1))
        path_cands = np.insert(X, indices, customer_id).reshape([-1, L+1])

        '''
        How to properly select here??
        '''
        costs = np.apply_along_axis(self.path_cost, 1, path_cands)
        if stochastic_choice:
            '''
            TODO: fix maybe so it becomes better. Right now it sucks.
            '''
            candidates = costs.argsort()[:3]
            if len(candidates) < 3:
                path_id = candidates[0]
            else:
                path_id = np.random.choice(candidates,
                                           p=(0.5, 0.3, 0.2))
        else:
            path_id = costs.argmin()

        return path_cands[path_id, :]

    def path_cost(self, path):
        '''
        Path is a sequence of customer (and depot) ids

        The path should include the depots as well.
        Example:
            path = [51, 12, 24, 51] where the depot is index 51.

        returns: a float
        '''

        p = np.array(path, dtype=np.int64)
        if self.distance_matrix is not None:
            retval = self.distance_matrix[p[:-1]-1, p[1:] - 1].sum()
            return retval

        spath = path.tostring()
        if spath in self.path_cost_dict:
            return self.path_cost_dict[spath]

        locs = np.vstack([
            self.customers[:, [0, 1, 2]],
            self.depots[:, [0, 1, 2]]
            ])
        xy_sequence = utils.get_by_ids(locs, path)[:, [1, 2]]

        cost = utils.euclidean_dist(xy_sequence)
        self.path_cost_dict[spath] = cost
        return cost

    def iter_all_paths(self, include_indices=False, include_depot=False):
        '''
        generator function that yields all paths in `self.routes`.

        include_indices, boolean. If True then will also yield the depot_id and
        index of the specific path as a tuple. Otherwise only returns the path.

        include_depot, boolean. If True then will include the depot index in the
        beginning and end of each path. E.g the path [5, 2, 1] becomes
        [51, 5, 2, 1, 52] where 52 is a specific depot index.
        '''
        for d in self.iter_depots():
            for i, each in enumerate(self.iter_paths(d, include_depot=include_depot)):
                if include_indices:
                    yield d, i, each
                else:
                    yield each

    def is_in_feasible_state(self, verbose=False):
        for d_idx, p_idx, path in self.iter_all_paths(include_indices=True, include_depot=True):
            carry_limit = self.depots[self.depots[:, 0] == d_idx].squeeze()[7]
            if self.capacity_requirement(path) > carry_limit:
                if verbose:
                    utils.cprint(f'{self} is [r]not in feasible state')
                    utils.cprint(f'found the path {path} having capacity {self.capacity_requirement(path)}, which is over the carry limit of {carry_limit}.')  # noqa

                return False

        return True

    def describe(self):
        all_customers = set()
        for d_idx in self.iter_depots():
            for i, path in enumerate(self.iter_paths(d_idx)):
                if len(set(path) & all_customers) > 0:
                    utils.cprint('[r]Warning: found customer at multiple paths')
                    print(set(path) & all_customers)

                all_customers = all_customers.union(set(path))
                demand = self.capacity_requirement(path)
                supply = self.depots[self.depots[:, 0] == d_idx].squeeze()[7]
                color = 'g' if demand <= supply else 'r'
                utils.cprint(f'[g]depot {d_idx}:[w] {path} [{color}]({demand} / {supply})')
        if len(all_customers) != 50:
            print(all_customers)
        

    '''
    Visualization stuff
    '''
    def visualize(self, ax=None, title=None):
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_prop_cycle(cycler('linestyle', '- -- : -.'.split()))

        locs = np.vstack([
            self.customers[:, [0, 1, 2]],
            self.depots[:, [0, 1, 2]]])

        ax.scatter(self.depots[:, 1], self.depots[:, 2], c='r', marker='*')
        ax.scatter(self.customers[:, 1], self.customers[:, 2], s=2*self.customers[:, 4], c='b', marker='.')
        for txt, x, y in zip(self.customers[:, 0], self.customers[:, 1], self.customers[:, 2]):
            txt = str(txt)
            ax.annotate(txt, (x, y), size=8)

        for color, depot_id in zip('c m y k r g b'.split(), self.iter_depots()):
            for path in self.iter_paths(depot_id, include_depot=True):
                xy_coords = utils.get_by_ids(locs, path)[:, [1, 2]]
                ax.plot(xy_coords[:, 0], xy_coords[:, 1], color, alpha=0.6)
        if title is not None:
            ax.set_title(title)
