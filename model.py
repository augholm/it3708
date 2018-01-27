import collections
import numpy as np
import clustering
import utils
import sample


class MDVRPModel():
    def __init__(self, customers, depots, verbose=True):
        '''
        depots: np.array of shape (N, 8) where there are N depots
        customers: np.array of shape (M, 5) where there are M customers.
        verbose: boolean, whether verbose output (prints) should be used or not
        '''
        self.depots = depots
        self.verbose = verbose
        self.customers = customers
        self.population = []

    def generate_initial_population(self, n=100):
        '''
        Makes the initial population and stores it in `self.population`.

        '''
        self.population = [sample.Individual(self.customers, self.depots)
                           for _ in range(n)]

    def fitness_score(self, individual):
        total_dist = 0

        locs = np.vstack([
            self.customers[:, [0, 1, 2]],
            self.depots[:, [0, 1, 2]]
            ])

        for depot_id in individual.iter_depots():
            for path in individual.iter_paths(depot_id, True):
                xy_sequence = utils.get_by_ids(locs, path)[:, [1, 2]]
                total_dist += utils.euclidean_dist(xy_sequence)

        return total_dist

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

model = MDVRPModel(customers, depots)
model.generate_initial_population()
for each in model.population:
    print(model.fitness_score(each))
