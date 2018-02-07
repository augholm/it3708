import numpy as np
import optimization
import utils
import sample
import matplotlib.pyplot as plt
plt.ion()


class MDVRPModel():
    def __init__(self, customers, depots, conf):
        '''
        depots: np.array of shape (N, 8) where there are N depots
        customers: np.array of shape (M, 5) where there are M customers.
        verbose: boolean, whether verbose output (prints) should be used or not
        '''
        self.depots = depots
        self.customers = customers
        self.population = []
        self.conf = conf

        X = np.vstack([
            customers,
            depots[:, :5]
            ])
        self.distance_matrix = utils.all_euclidean_dist(X[:, [1, 2]])

        if conf['generate_initial_population']:
            population_size = utils.get_population_size(conf, 0)
            self.generate_initial_population(population_size)

    def generate_initial_population(self, n, add_path_cost_dict=True):
        '''
        Makes the initial population and stores it in `self.population`.

        '''
        self.path_cost_dict = {}
        self.population = [sample.Individual(self.customers, self.depots, path_cost_dict=self.path_cost_dict, distance_matrix=self.distance_matrix)
                           for _ in range(n)]

    def fitness_score(self, individual):
        total_dist = 0

        for depot_id in individual.iter_depots():
            for path in individual.iter_paths(depot_id, include_depot=True):
                total_dist += individual.path_cost(path)

        return total_dist

    def selection(self, p=0.8):
        '''
        A tournament selection strategy is used and we use elitist selection.

        k: int, population size to consider in pool
        n: int, population size used for breeding
        p: float, probability of choosing best individual out of population, otherwise
        random is chosen

        returns: a list of size `n` of individuals
        '''
        k = self.conf['tournament_size']
        n = self.conf['breeding_pool']
        p = self.conf['tournament_best_prob']
        population_size  = utils.profile_value(self.conf['population_profile'], 0, self.conf['num_generations'])

        if k > population_size: k = population_size
        if n > population_size: n = population_size

        L = []
        if self.conf['weighted_selection']:
            costs = np.array([1/self.fitness_score(each) for each in self.population])
            weights = costs / costs.sum()
        else:
            weights = None

        for _ in range(n):
            X = np.random.choice(self.population, k, replace=False, p=weights)
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
        import copy
        child = copy.deepcopy(p1)

        depot_id = utils.choose_random_depot(p1, n_depots=1)
        path_1 = utils.choose_random_paths(p1, depot_id=depot_id)[0]
        path_2 = utils.choose_random_paths(p2, depot_id=depot_id)[0]

        for c_id in path_1:
            child.move_customer(c_id, intra_depot=True, stochastic_choice=True)

        for c_id in path_2:
            child.move_customer(c_id, intra_depot=True, stochastic_choice=True)
        return child

        return copy.deepcopy(np.random.choice([p1, p2]))
        #Randomly select depot x in set of depots to undergo reproduction
        depot = np.random.choice(self.depots[:, 0], 1)

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

    def mutation(self, individual, intra_depot=True, n_mutations=1):
        '''
        Apply a mutation. Randomly choose between three different kinds of mutations.

        individual: The individual that the mutation will be applied to
        intra_depot: boolean, if True then the mutation will only affect tours within the
        same depot. Otherwise will consider all tours.

        '''

        def reversal_mutation():
            '''
            Finds a random path. Then reverses a subpath in that path.

            Example:
                6 9 5 4 7 2
                reverse around index 1 and before index 4:
                --> 6 4 5 9 7 2
            '''

            try:
                depot_id, path_id, path = utils.choose_random_paths(
                    individual, n=1, min_length=2, include_indices=True,
                    depot_id='random' if intra_depot else 'all')[0]
            except utils.NoPathFoundException:
                utils.cprint('[y]WARNING: found no path in swapping')
                return

            a, b = np.random.choice(range(len(path)), 2, replace=False)
            a, b = min(a, b), max(a, b)
            path[a:b] = path[a:b][::-1]
            individual.tours[depot_id][path_id] = path

        def single_customer_rerouting():
            '''
            From the paper:

            "Re-routing involves randomly selecting one customer, and removing
            that customer from the existing route. The customer is then
            inserted in the best feasible insertion location within the entire
            chromosome. This involves computing the total cost of insertion at
            every insertion locale, which finally re-inserts the customer in
            the most feasible location."

            returns: None, but moves a (random) customer to a different route.
            '''

            customer_id = np.random.choice(individual.customers[:, 0])
            individual.move_customer(customer_id, intra_depot=intra_depot, stochastic_choice=False)

        def swapping():
            '''
            From the paper:

            This simple mutation operator selects two random routes and swaps one
            randomly chosen customer from one route to another
            '''

            if intra_depot:
                depot_id = utils.choose_random_depot(individual,
                                                     min_num_paths=2)
            else:
                depot_id = 'all'

            try:
                (d_idx1, p_idx1, p1), (d_idx2, p_idx2, p2) = utils.choose_random_paths(
                    individual, n=2, min_length=2, depot_id=depot_id, include_indices=True, include_depot=False)
            except utils.NoPathFoundException:
                print('found no path!! in swapping')
                return

            # p1 = np.array(p1, np.int64)
            # p2 = np.array(p2, np.int64)

            _p1 = p1
            _p2 = p2
            c1 = np.random.choice(p1)  # TODO: fix.
            c2 = np.random.choice(p2)
            #c2 = p2[self.distance_matrix[c1-1, p2-1].argmin()]

            '''
            Before calling optimally insert we need to prepend and append the
            depot.
            '''
            p1 = np.hstack([d_idx1, utils.delete_by_value(p1, c1), d_idx1])
            p2 = np.hstack([d_idx2, utils.delete_by_value(p2, c2), d_idx2])

            def insert_randomly(arr, elem):
                n = len(arr)
                return np.insert(arr, np.random.choice(range(n + 1)), elem)

            ap1 = individual.optimally_insert(p1, c2, stochastic_choice=True)
            ap2 = individual.optimally_insert(p2, c1, stochastic_choice=True)

            capacity_limit_1 = individual.depots[individual.depots[:, 0] == d_idx1].squeeze()[7]
            capacity_limit_2 = individual.depots[individual.depots[:, 0] == d_idx2].squeeze()[7]

            X = np.vstack([
                individual.customers[:, [0, 1, 2]],
                individual.depots[:, [0, 1, 2]]
                ])

            if (individual.capacity_requirement(ap1) <= capacity_limit_1 and
                    individual.capacity_requirement(ap2) <= capacity_limit_2):
                import copy
                ap1_ = copy.deepcopy(ap1)
                ap2_ = copy.deepcopy(ap2)
                ap1 = new_path = self._solve_tsp(ap1[1:-1], d_idx1)
                ap2 = self._solve_tsp(ap2[1:-1], d_idx2)
                individual.tours[d_idx1][p_idx1] = ap1
                individual.tours[d_idx2][p_idx2] = ap2

        def one_in_one_out():
            '''
            Choose a customer. Then, try to see if we can remove one or more
            of the path on the other side and add new that are close to it.
            '''
            if intra_depot:
                depot_id = utils.choose_random_depot(individual,
                                                     min_num_paths=2)
            else:
                depot_id = 'all'

            try:
                (d_idx, p_idx, path)  = utils.choose_random_paths(
                    individual, n=2, min_length=5, depot_id=depot_id, include_indices=True, include_depot=False)[0]
            except utils.NoPathFoundException:
                # utils.cprint('[r]Warning:[w] Found no path in one_in_one_out mutation')
                return

            capacity_limit = self.depots[self.depots[:, 0] == d_idx].squeeze()[7]
            c = np.random.choice(path)
            D = self.distance_matrix
            # sorted = np.asarray(np.argsort(D[c-1, p-1])).squeeze()
            radial_distance = utils.matrix_row_to_array(np.argsort(D[c-1, :]))
            radial_distance += 1
            in_path = []
            outside_path = []
            for each in radial_distance:
                if each in self.depots[:, 0]:
                    continue
                if each in path:
                    in_path.append(each)
                else:
                    outside_path.append(each)

            intra = np.array(in_path)[-3:]
            outside = np.array(outside_path)[:5]


            L = []
            X = np.vstack(
                    [self.customers, self.depots[:, :5]])

            for leave in intra:
                candidate_path = utils.delete_by_value(path, leave)
                remaining_space = capacity_limit - individual.capacity_requirement(candidate_path)
                for enter in outside:
                    demand = individual.capacity_requirement([enter])
                    if demand > remaining_space:
                        continue

                    path_with_depot = np.hstack((d_idx, candidate_path, d_idx))
                    P = individual.optimally_insert(path_with_depot, enter, stochastic_choice=False)
                    xy_locs = utils.get_by_ids(X, P)[:, [1, 2]]
                    distance = utils.euclidean_dist(xy_locs)

                    L.append((distance, P[1:-1], leave, enter))

            # import ipdb; ipdb.set_trace()
            if len(L) == 0:
                return
            min_score, new_path, leave_customer, enter_customer = min(L, key=lambda x: x[0])

            # import ipdb; ipdb.set_trace()
            # individual.remove_customer(leave_customer)
            individual.move_customer(leave_customer, intra_depot=False)  # now move this fuck
            # individual.tours[d_idx][p_idx] = new_path
            new_path = self._solve_tsp(new_path, d_idx)
            individual.tours[d_idx][p_idx] = new_path

        def local_opt():
            X = np.vstack([
                individual.customers[:, [0, 1, 2]],
                individual.depots[:, [0, 1, 2]]
                ])
            (d_idx, p_idx, path) = utils.choose_random_paths(
                individual, n=1, min_length=3, depot_id='random', include_indices=True)[0]

            new_path = self._solve_tsp(path, d_idx)
            individual.tours[d_idx][p_idx] = new_path

        for _ in range(n_mutations):
            id = np.random.choice(
                range(7),)
            if id == 0:
                single_customer_rerouting()
            if id == 1:
                reversal_mutation()
            if id == 2:
                pass
            if id == 3:
                local_opt()
            if id == 4:
                single_customer_rerouting()
            if id == 5:
                one_in_one_out()
                local_opt()
            if id == 6:
                one_in_one_out()

    def run_step(self, generation_step):
        '''
        TODO: fix parameter sizes and stuff
        e.g the population size
        '''

        individuals = self.selection()
        new_offspring = []

        current_population_size = utils.profile_value(self.conf['population_profile'], generation_step, self.conf['num_generations'])

        n_children = max(1, int(self.conf['birth_rate'] * current_population_size))
        intra_depot = False if generation_step % self.conf['extra_depot_every'] == 0 else True

        n_mutations = utils.profile_value(
            self.conf['mutation_profile'], generation_step, self.conf['num_generations'])

        while True:
            p1, p2 = np.random.choice(individuals, 2, replace=False)
            offspring = self.create_offspring(p1, p2)
            # if not offspring.is_in_feasible_state():
            #     import ipdb; ipdb.set_trace()
            #     pass
            # self.mutation(offspring, intra_depot=intra_depot, n_mutations=n_mutations)
            # #except ValueError:
            #     # could occur if multiple mutations..
            # #utils.cprint('[r]whoopsie, this individual became a retard :S')
            if offspring.is_in_feasible_state():
                new_offspring.append(offspring)
            else:
                pass
            if len(new_offspring) == n_children:
                break

        
        X = np.concatenate([new_offspring, self.population])

        indices = np.argsort([self.fitness_score(e) for e in self.population])
        D = np.array(self.population)
        X = np.concatenate([D[indices], new_offspring])

        ps = current_population_size
        best = indices[:50]
        other = new_offspring
        # other = np.random.choice(indices[10:], ps - 10)
        # best = indices[:ps//2]
        # other = np.random.choice(indices[ps//2:], ps//2)
        # new_generation = X[np.concatenate((best, other))]
        new_generation = X
        for each in new_generation:
            if np.random.choice([True, False], p=(0.2, 0.8)):
                self.mutation(each, intra_depot=False)

        self.population = new_generation

    def evolve(self, visualize_step=None):
        '''
        repeatedly calls `self.run_step` and eventually plots progres.
        '''
        min_scores = []
        mean_scores = []
        steps = []

        if visualize_step is not None:
            if len(plt.gcf().get_axes()) < 4:
                fig, axes = plt.subplots(2, 2)
                axes = axes.flatten()
            else:
                fig, axes = plt.gcf(), plt.gcf().get_axes()

        for i in range(self.conf['num_generations']):
            self.run_step(i)

            # TODO: plotting could probably be fixed in its own function somehwere.
            data = [(each, self.fitness_score(each)) for each in self.population]
            fittest, score = min(data, key=lambda x: x[1])
            scores = [each[1] for each in data]
            ms = np.mean(scores)
            min_scores.append(min(scores))
            mean_scores += scores
            steps += [i] * len(scores)
            if i % 5 == 0:
                sz = utils.get_population_size(self.conf, i)
                utils.cprint(f'min score in step [y]{i}[w] is [y]{score}[w] and mean is [y]{ms}[w]. Population size is [y]{sz}')

            if visualize_step is not None and i % visualize_step == 0:
                fittest.visualize(ax=axes[0], title=i)

                axes[1].cla()
                axes[1].plot(min_scores, label='min score')
                #axes[1].plot(mean_scores, label='mean score')
                axes[1].scatter(steps, mean_scores, alpha=0.1, cmap='viridis')
                axes[1].legend()
                plt.pause(0.05)

                axes[2].cla()
                X = [self.fitness_score(e) for e in self.population]
                axes[2].hist(X,bins=100)
                axes[2].set_title('score distribution')

    def _solve_tsp(self, seq, depot_idx):
        '''
        Simple wrapper around optimization.solve_tsp
        '''
        X = np.vstack([
            self.customers[:, [0, 1, 2]],
            self.depots[:, [0, 1, 2]]
            ])

        depot_and_customer_locs = X[np.hstack((depot_idx-1, seq-1))][:, [1, 2]]
        try:
            indices = optimization.solve_tsp(depot_and_customer_locs)
        except:
            # in case not able to solve..
            utils.cprint('[y]could not solve tsp -- no convex hull found')
            return seq

        # we do not want to include the depot in our seq. Delete it. We also have to substract by 1
        indices = utils.delete_by_value(indices, 0) - 1
        # now indices is just a permutation of the seq

        return seq[indices]
