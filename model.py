import matplotlib.pyplot as plt
import numpy as np

import utils
import sample2


class MDVRPModel():
    def __init__(self, customers, depots, conf):
        '''
        depots: np.array of shape (N, 8) where there are N depots
        customers: np.array of shape (M, 5) where there are M customers.
        verbose: boolean, whether verbose output (prints) should be used or not
        '''
        self.conf = conf
        self.population = []

        depot_ids = depots[:, 0] - 1
        customer_info = customers[:, [1, 2, 4]]
        depot_info = depots[:, [1, 2, 7]]

        self.X = np.vstack((
            np.array(customer_info, dtype=np.int64),
            np.array(depot_info, dtype=np.int64)))

        self.depots = np.array(depot_ids, dtype=np.int64)
        self.D = utils.all_euclidean_dist(self.X[:, [0, 1]])

        if conf['generate_initial_population']:
            population_size = utils.get_population_size(conf, 0)
            self.generate_initial_population(population_size)

    def generate_initial_population(self, n):
        L = []
        for _ in range(n):
            L.append(sample2.Individual(self.X, self.depots, self.D, initialize=True))

        self.population = L


    def selection(self):
        '''
        '''
        k = self.conf['tournament_size']
        n = self.conf['breeding_pool']
        p = self.conf['tournament_best_prob']
        population_size  = utils.profile_value(self.conf['population_profile'], 0, self.conf['num_generations'])
        if k > population_size: k = population_size
        if n > population_size: n = population_size
        scores = np.array([each.fitness_score() for each in self.population])
        L = []

        candidates = np.arange(len(self.population))
        while len(L) < n:
            if k > len(candidates):
                I = candidates
            else:
                I = np.random.choice(candidates, k, replace=False)
            if np.random.uniform(0, 1) <= p:
                idx = I[scores[I].argmin()]
            else:
                idx = np.random.choice(I)
            L.append(self.population[idx])
            candidates = utils.delete_by_value(candidates, idx)

        return L

    def create_offspring(self, mother, father):
        import copy
        child = copy.deepcopy(father), copy.deepcopy(mother)

        c1 = copy.deepcopy(father)
        c2 = copy.deepcopy(mother)

        d_idx = np.random.choice(father.depots)
        p1 = utils.choose_random_paths(c1, depot_id=d_idx, min_length=1)[0]
        p2 = utils.choose_random_paths(c2, depot_id=d_idx, min_length=1)[0]

        for c in p1:
            c2.move_customer(c, intra_depot=False)
        for c in p2:
            c1.move_customer(c, intra_depot=False)

        return c1

        pass

    def mutation(self, individual, intra_depot=True):
        i = individual
        p, d_idx, p_idx = utils.choose_random_paths(i, min_length=3, include_indices=True, include_depot=False)[0]
        if p[0] == p[-1]:
            import ipdb; ipdb.set_trace()
        def swap(p, d_idx, p_idx):
            a, b = np.random.choice(range(len(p)), 2, replace=False)
            a, b = min(a, b), max(a, b)
            p[a:b] = p[a:b][::-1]

            i.update_path(p, d_idx, p_idx)
            pass

        def tsp(path, d_idx, p_idx):
            p_with_depot = np.hstack((d_idx, path, d_idx))
            path = i.solve_tsp(p_with_depot)
            i.update_path(path, d_idx, p_idx)

        functions = [swap, tsp]
        call = np.random.choice(functions)
        call(p, d_idx, p_idx)

    def run_one_step(self, generation_step):
        current_population_size = utils.profile_value(self.conf['population_profile'], generation_step, self.conf['num_generations'])
        n_children = max(1, int(self.conf['birth_rate'] * current_population_size))
        intra_depot = False if generation_step % self.conf['extra_depot_every'] == 0 else True

        L = []
        for _ in range(n_children):
            p1, p2 = np.random.choice(self.population, 2, replace=False)
            offspring = self.create_offspring(p1, p2)
            L.append(offspring)

        L = np.concatenate((L, self.population))
        scores = np.argsort([each.fitness_score() for each in L])
        for each in self.population:
            if np.random.choice((True, False), p=(0.2, 0.8)) == True:
                self.mutation(each)
        scores = np.argsort([each.fitness_score() for each in L])
        self.population = L[scores[:current_population_size]]

    def evolve(self, visualize_every=None):
        for t in range(1, self.conf['num_generations'] + 1):
            self.run_one_step(t)
            if visualize_every != 0 and visualize_every is not None and t % visualize_every == 0:
                self.visualize(t)

    def visualize(self, step, ax=None):
        scores = np.array([each.fitness_score() for each in self.population])
        best_individual = self.population[scores.argmin()]

        if ax is None and len(plt.get_fignums()) == 0:
            fig, (ax0, ax1) = plt.subplots(1, 2)
        else:
            ax0, ax1 = plt.gcf().get_axes()
        
        best_individual.visualize(ax=ax0, title='best fit')
        utils.cprint(f'step [y]{step}[w], mean score is [y]{scores.mean()}[w] and best is {best_individual.fitness_score()}')
        plt.pause(0.05)


