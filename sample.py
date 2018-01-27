import numpy as np
import utils
import collections
import clustering


class Individual():
    def __init__(self, customers, depots, initialize=True):
        '''
        customers: np.array representing the customers
        depots: np.array representing the depots
        initialize: boolean, if True will call self.generate_initial_state
        '''
        self.customers = customers
        self.depots = depots
        self.tours = collections.defaultdict(lambda: [])
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

        def split_to_paths(sequence, carry_limit):
            '''
            sequence: list of customer indices

            returns: a list of of subsequences, each with total capacity below
            maximum allowed
            '''
            all_tours = []
            subtour = []
            for entry in sequence:
                if self.capacity_requirement(subtour + [entry]) < carry_limit:
                    subtour.append(entry)
                else:
                    all_tours.append(subtour)
                    subtour = []
            return all_tours + subtour

        for dept_id, carry_limit in self.depots[:, [0, 7]]:
            assignment = self.customers[Y == dept_id][:, 0]
            self.tours[dept_id] = split_to_paths(np.random.permutation(assignment),
                                                 carry_limit)

    def iter_paths(self, depot_id, include_depot_at_start_and_end=False):
        '''
        include_depot_at_start_and_end: boolean, if true then also include the
        depot id in the beginning / end

        Usage:
            I = Individual(customers, depots)
            for tour in I.iter_paths(52, include_depot_at_start_and_end=False):
                >> [42, 65, 24]
                >> [38, 66, 12]
                >> ...

            for tour in I.iter_paths(52, include_depot_at_start_and_end=True):
                >> [52, 42, 65, 24, 52]
                >> [52, 38, 66, 12, 52]
                >> ...

        '''
        for tour in self.tours[depot_id]:
            if not utils.is_iterable(tour):
                tour = [tour]
            if include_depot_at_start_and_end:
                yield [depot_id] + tour + [depot_id]
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

        C = self.customers
        X = C[np.array(tour) - 1][:, 4]
        return X.sum()
