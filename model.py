import matplotlib.pyplot as plt
import ipdb
import numpy as np

import utils
import sample


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
        self.D = np.array(utils.all_euclidean_dist(self.X[:, [0, 1]]))
        self.N = utils.find_neighbourhood(self.D, conf['n_neighbours'], ignore_indices=self.depots)

        self.best_score = np.inf
        self.best_count = 0
        self.mean_violations = []
        self.penalty_multiplier = 1

        if conf['generate_initial_population']:
            population_size = utils.get_population_size(conf, 0)
            self.population = self.generate_initial_population(population_size)

    def generate_initial_population(self, n):
        L = []
        for _ in range(n):
            L.append(sample.Individual(self.X, self.depots, self.D, self.N, initialize=True))

        return L


    def selection(self):
        '''
        '''
        k = self.conf['tournament_size']
        n = self.conf['breeding_pool']
        p = self.conf['tournament_best_prob']
        population_size  = utils.profile_value(self.conf['population_profile'], 0, self.conf['num_generations'])
        if k > population_size: k = population_size
        if n > population_size: n = population_size
        scores = np.array([each.fitness_score(self.penalty_multiplier) for each in self.population])
        # scores = np.array([each.fitness_score(self.penalty_multiplier) for each in self.population])
        # scores = np.array([each.total_violation() for each in self.population])
        # if np.random.choice((0, 1), p=(0.01, 0.99)):
        #     import ipdb; ipdb.set_trace()
        L = []

        candidates = np.arange(len(self.population))
        weights = (1/scores) / (1/scores).sum()
        while len(L) < n:
            if k >= len(candidates):
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

        # partition the depot assignments in sets
        data = np.random.permutation(np.array(self.depots, copy=True))
        delim0, delim1 = np.random.choice(np.arange(data.shape[0]), size=2, replace=False)
        delim0, delim1 = min(delim0, delim1), max(delim0, delim1)
        A1 = data[:delim0]
        A2 = data[delim0:delim1]
        A_mix = data[delim1:]

        child = sample.Individual(self.X, self.depots, self.D, self.N, initialize=False)
        for d_idx in A1:
            for p, c, cap, p_idx in father.iter_paths(d_idx, yield_depot_idx=False, yield_path_idx=True):
                child.update_path(p, d_idx, p_idx, check_feasibility_after=False)
        for d_idx in A_mix:
            for p, c, cap, p_idx in father.iter_paths(d_idx, yield_depot_idx=False, yield_path_idx=True):
                if len(p) >= 2:
                    alpha, beta = np.random.choice(np.arange(len(p)), 2, replace=False)
                    alpha, beta = min(alpha, beta), max(alpha, beta)
                    child.update_path(p[alpha:beta+1], d_idx, p_idx, check_feasibility_after=False)
                else:
                    child.update_path(p, d_idx, p_idx, check_feasibility_after=False)

        served_customers = utils.list_served_customers(child)
        unserved_customers = np.setdiff1d(
            np.arange(self.X.shape[0] - self.depots.shape[0]),
            served_customers)

        for d_idx in np.random.permutation(np.concatenate((A2, A_mix))):
            for p, c, cap, p_idx in mother.iter_paths(d_idx, yield_depot_idx=False, yield_path_idx=True):
                p_ = np.setdiff1d(p, served_customers)
                unserved_customers = np.setdiff1d(unserved_customers, p_)
                if d_idx in A_mix:
                    for c in p_:
                        child.optimally_insert_customer(c)
                else:
                    child.update_path(p_, d_idx, p_idx, check_feasibility_after=False)

        for each in unserved_customers:
            child.optimally_insert_customer(each)
            served = utils.list_served_customers(child)
            pass


        before = child.fitness_score()
        if np.random.choice((0,1), p=(0.8, 0.2)):
            child.RI(self.penalty_multiplier)
        if child.average_capacity_infeasibility() > 0 and np.random.choice((0,1)):
            child.repair()
        after = child.fitness_score()

        # child.route_improvement()
        # child.route_improvement()
        # self.mutation(child, intra_depot=True)
        return child

    def mutation(self, individual, intra_depot):
        i = individual
        p, d_idx, p_idx = utils.choose_random_paths(i, min_length=3, include_indices=True, include_depot=False)[0]
        if p[0] == p[-1]:
            import ipdb; ipdb.set_trace()

        def reversal(p, d_idx, p_idx):
            a, b = np.random.choice(range(len(p)), 2, replace=False)
            a, b = min(a, b), max(a, b)
            p[a:b] = p[a:b][::-1]

            i.update_path(p, d_idx, p_idx)

        def tsp(path, d_idx, p_idx):
            p_with_depot = np.hstack((d_idx, path, d_idx))
            new_path = i.solve_tsp(p_with_depot)
            if i.cost(p_with_depot) - 2 > i.cost(new_path):
                i.update_path(new_path, d_idx, p_idx)

        def single_customer_rerouting(path, d_idx, p_idx):
            i.move_customer(np.random.choice(path), depot_id=None if intra_depot else d_idx)

        def improvement():
            individual.route_improvement(penalty=self.penalty_multiplier)

        def swap(path, d_idx, p_idx):
            try:
                if intra_depot:
                    d_idx = np.random.choice(i.depots)
                    x = utils.choose_random_paths(i, n=2, min_length=1, depot_id=d_idx, include_indices=True, include_depot=False)
                else:
                    x = utils.choose_random_paths(i, n=2, min_length=3, include_indices=True, include_depot=False)
            except utils.NoPathFoundException:
                # utils.cprint('[y]warning: no path found; returning in swap mutation')
                return


            p1, d1_idx, p1_idx = x[0]
            p2, d2_idx, p2_idx = x[1]
            i.move_customer(np.random.choice(p1), depot_id=d2_idx)
            i.move_customer(np.random.choice(p2), depot_id=d1_idx)

        # functions = [single_customer_rerouting, lambda x,y,z: improvement()]
        functions = [tsp]
        call = np.random.choice(functions)
        call(p, d_idx, p_idx)

    def run_one_step(self, generation_step):
        population_size = utils.profile_value(self.conf['population_profile'], generation_step, self.conf['num_generations'])
        n_children = max(1, int(self.conf['birth_rate'] * population_size))
        intra_depot = False if generation_step % self.conf['extra_depot_every'] == 0 else True

        L = []
        for _ in range(n_children):
            pool = self.selection()
            p1 = np.random.choice(pool)
            similarities = [utils.similarity(p1, each) for each in self.population]
            p2 = self.population[np.argmin(similarities)]

            p1, p2 = np.random.choice(pool, 2, replace=False)
            offspring = self.create_offspring(p1, p2)
            L.append(offspring)

        L = np.concatenate((L, self.population))

        for each in L:
            if each.total_violation() > 0 and np.random.choice((0, 1), p=(0.8, 0.2)):
                each.repair()
            if np.random.choice((0, 1), p=(.9, .1)):
                self.mutation(each, True)

        scores = np.argsort([each.fitness_score(self.penalty_multiplier) for each in L])
        if scores[0] != self.best_score:
            self.best_score = scores[0]
        self.best_count += 1
        acis = [each.average_capacity_infeasibility() for each in self.population]
        self.mean_violations.append(np.mean(acis))
        if len(self.mean_violations) > 20:
            self.mean_violations = self.mean_violations[-20:]

        '''
        Penalty parameter adjustment
        '''
        if generation_step % 4 == 0:

            prop_feasible = np.mean(np.array(self.mean_violations) == 0)
            if prop_feasible - 0.05 > self.conf['fraction_feasible_population']:
                # allow more violations -- except if the multplier is low
                if not self.penalty_multiplier <= 1:
                    self.penalty_multiplier *= 0.85
            elif prop_feasible + 0.05 < self.conf['fraction_feasible_population']:
                # allow fewer violations
                self.penalty_multiplier *= 1.2

        self.best = scores[0]
        if self.best_count >= 100:
            print('hey now I change')
            old_population = L[scores[:population_size // 4]]
            new_population = self.generate_initial_population(population_size - len(old_population))
            self.population = np.concatenate((old_population, new_population))
            self.best_count = 0
        else:
            # ipdb.set_trace()
            self.population = L[scores[:population_size]]

    def evolve(self, visualize_every=None):
        for t in range(1, self.conf['num_generations'] + 1):
            self.run_one_step(t)
            if visualize_every != 0 and visualize_every is not None and t % visualize_every == 0:
                self.visualize(t)

    def visualize(self, step, ax=None):
        scores = np.array([each.fitness_score() for each in self.population])
        best_individual = self.population[scores.argmin()]
        best_fit = best_individual.fitness_score(0)
        feasibility = best_individual.total_violation()

        if ax is None and len(plt.get_fignums()) == 0:
            fig, (ax0, ax1) = plt.subplots(1, 2)
        else:
            ax0, ax1 = plt.gcf().get_axes()

        best_individual.visualize(ax=ax0, title=f'best fit ({best_fit:6.2f})')
        utils.cprint(f'step [y]{step}[w], best score: [y]{best_individual.fitness_score(0):.5} ({feasibility})[w] mean score: [y]{scores.mean():.5}. [w]Penalty multiplier is[y] {float(self.penalty_multiplier):.3}')  # noqa
        plt.pause(0.05)
