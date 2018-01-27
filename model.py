import collections
import numpy as np
import clustering
import utils


class MDVRPModel():
    def __init__(self, depots, customers, verbose=True):
        '''
        depots: np.array of shape (N, 8) where there are N depots
        customers: np.array of shape (M, 5) where there are M customers.
        verbose: boolean, whether verbose output (prints) should be used or not
        '''
        self.depots = depots
        self.verbose = verbose
        self.customers = customers

    def generate_initial_population(self, n=100):
        D, C = self.depots, self.customers

        n_depots = D.shape[0]
        n_customers = C.shape[0]

        chromosomes = []

        Y = clustering.nearest_depot(self.depots, self.customers)

        for _ in range(n):
            L = []
            # randomly assign to different groups
            I = np.random.choice(np.arange(n_depots), n_customers, replace=True)  # noqa
            for i in range(n_depots):
                customer_ids = C[Y == i, 0]
                L.append(np.random.permutation(customer_ids))
            chromosomes.append(self.route_scheduler(L))

        self.population = chromosomes
        if self.verbose:
            print('population created')

    def route_scheduler(self, chromosome):
        '''
        Direct (and naive) implementation of the route scheduler as described in
        ch. 3.3 in Berman & Hanshar.
        (https://link.springer.com/content/pdf/10.1007/978-3-540-85152-3_4.pdf)

        Returns a list where each entry in the list is a list of routes. Each
        route is split with a zero. Example:

        Example:
            return L = [L0, L1, L2]
                L0 == [0, 13, 18, 44, 0, 15, 45, 4, 25 0]
                which means it starts from depot (0), then goes to 13 -> 18 -> 44
                -> 0 (back to depot) -> 15 -> 45 -> 4 -> 25 -> 0 (back to depot).
                Notice that 13 means customer_id == 13 (not the index!)
                To access customer no. 13, use  utils.get_by_ids(customers, 13)

                L1 corresponds to the routes for the depots at index 1 (i.e depots[1])
                and similarly for L2.

        returns a list
        '''
        S, C, D = chromosome, self.customers, self.depots

        '''
        Stage 1
        '''
        routes = collections.defaultdict(lambda: [])
        for d, each in zip(D, S):
            depot_id, max_load = d[0], d[7]
            while len(each) > 0:
                customer_ids = []
                # does this include multiple tours or a single tour??
                while self.tour_required_capacity(customer_ids) < max_load and len(each) > 0:
                    new_customer = each[0]
                    each = each[1:]
                    customer_ids.append(new_customer)
                    #customer_ids = [depot_id] + customer_ids + [depot_id]
                routes[depot_id].append(customer_ids)  # maximal path found -- break

        '''
        Stage 2
        '''
        for depot_id, tours in routes.items():
            trc = lambda x: self.tour_required_capacity(x)  # noqa
            max_load = utils.get_by_ids(D, depot_id)[7]

            if len(tours) == 1:
                continue

            for i, (t1, t2) in enumerate(zip(tours, tours[1:])):
                t1_ = t1[:-1]
                t2_ = [t1[-1]] + t2

                '''
                If it turns out that removing the last city on t1 and adding that city
                at the beginning of t2 does not violate capacity constraints AND
                it is an overall shorter path compared to t1 and t2 alone, then we
                move the last city of t1 over to t2 and update the `tours` variable
                accordingly. Otherwise do nothing.
                '''
                if trc(t2_) <= max_load and trc(t1_) + trc(t2_) < trc(t1) + trc(t2):
                    print('stage 2 actually happened')  # TODO: remove
                    tours[depot_id][i] = t1_
                    tours[depot_id][i+1] = t2_

        def to_chromosome(routes):
            L = []
            delimiter = 0
            for k, v in routes.items():
                sub_L = [delimiter]
                for each in v:
                    sub_L = sub_L + each + [delimiter]
                L.append(sub_L)
            return L

        return to_chromosome(routes)

    def fitness_score(self, chromosome):
        total_score = 0

        # only extract the indices and their positions
        locs = np.vstack([
            self.customers[:, [0, 1, 2]],
            self.depots[:, [0, 1, 2]]
            ])
        for depot, tour in zip(self.depots, chromosome):
            tour = np.array(tour)
            depot_id = depot[0]

            np.place(tour, tour == 0, depot_id)  # replace the 0's with the actual ID
            X = utils.get_by_ids(locs, tour)[:, [1, 2]]
            score = utils.euclidean_dist(X)
            total_score += score
        return total_score

    def tour_required_capacity(self, customer_ids):
        '''
        For a given tour (represented by the customer indices), finds the
        total demand by all customers on that tour. The demand for each
        customer is found by the 5th column in `customers`.

        Example:
            customer_ids = array([1, 5, 8])
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
        if len(customer_ids) == 0:
            return 0

        C = self.customers
        X = C[np.array(customer_ids) - 1][:, 4]
        return X.sum()

    def selection(k=3, n=10, p=0.8):
        '''
        A tournament selection strategy is used and we use elitist selection.

        k: int, population size to consider in pool
        n: int, population size used for breeding
        p: float, probability of choosing best individual out of population, otherwise
        random is chosen

        returns: a list of size `n` of individuals
        '''
        L = []
        for _ in range(n):
            X = np.random.choice(self.population, k, replace=False)
            scores = [(self.fitness_score(each), each) for each in X]
            if np.random.uniform(0, 1) <= p:
                L.append(min(scores, key=lambda x: x[0])[1])
            else:
                L.append(np.random.choice(X))
        return L

    def create_offspring(self, p1, p2):
        '''
        p1: individual, representing first parent
        p2: individual, representing second parent

        returns: an individual
        '''


        #Randomly select depot x in set of depots to undergo reproduction
        selected_depot = np.random.choice(self.depots, 1)

        #Randomly select one route from each parent
        p1_routes = p1[selected_depot]
        p1_routes = p2[selected_depot]

        #Randomly select a route in given depot from each parent




