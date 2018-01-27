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

<<<<<<< Updated upstream

        #Randomly select depot x in set of depots to undergo reproduction
        depot = np.random.choice(self.depots, 1)

        #Randomly select one route from each parent, p1_
        p1_routes = p1.iter_paths(depot)
        p2_routes = p2.iter_paths(depot)

        #randomly select a route in given depot from each parent
        #routes now contain customers
        p1_random_route = np.random.choice(p1_routes, 1)
        p2_random_route = np.random.choice(p2_routes, 1)

        #From p1, remove the customers belonging to the randomly selected
        #route in p2

        new_p1_rotes = self.delete_customer(p1_routes, p2_routes)

        #From p1, remove the customers belonging to the randomly selected
        #route in p2

        new_p2_routes = self.delete_customer(p2_routes, p1_routes)




    #TODO:change to generic names
    def delete_customer(self, p1_routes, p2_routes):
        for customer in p2_routes:
            np.delete(p1_routes, np.argwhere(p1_routes == customer))
        return p1_routes

    #TODO:change to generic names
    def insert_customer(self, new_p1_routes, p2_routes, prob):


        if np.random.uniform(0,1) <= prob:
            insertion_costs = []
            for location in range(len(new_p1_routes)):
                insertion_costs.append((new_p1_routes.path_cost(location, new_p1_routes),location))

            best_location = min(insertion_costs, key=insertion_costs[0])[1]
            np.insert(new_p1_routes, best_location)

        else:
            np.insert(new_p1_routes, np.random.uniform(0,len(new_p1_routes),Size=int))

        return new_p1_routes






model = MDVRPModel(customers, depots)
model.generate_initial_population()
for each in model.population:
    print(model.fitness_score(each))
=======
        C = self.customers
        X = C[np.array(customer_ids) - 1][:, 4]
        return X.sum()

>>>>>>> Stashed changes
