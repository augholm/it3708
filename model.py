import numpy as np
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

        TODO: add config support so we can add tweakings to the model
        '''
        self.depots = depots
        self.customers = customers
        self.population = []
        self.conf = conf

        if conf['generate_initial_population']:
            self.generate_initial_population(conf['population_size'])

    def generate_initial_population(self, n, add_path_cost_dict=True):
        '''
        Makes the initial population and stores it in `self.population`.

        '''
        self.path_cost_dict = {}
        self.population = [sample.Individual(self.customers, self.depots, path_cost_dict=self.path_cost_dict)
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

    def mutation(self, individual, intra_depot=True, degree=1):
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
                print('found no path')
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
            individual.move_customer(customer_id, intra_depot=intra_depot)

        def swapping():
            '''
            From the paper:

            This simple mutation operator selects two random routes and swaps one
            randomly chosen customer from one route to another
            '''

            depot_id = utils.choose_random_depot(individual,
                                                 min_num_paths=2)

            try:
                (d_idx1, p_idx1, p1), (d_idx2, p_idx2, p2) = utils.choose_random_paths(
                    individual, n=2, depot_id=depot_id, include_indices=True, include_depot=False)
            except utils.NoPathFoundException:
                print('found no path!! in swapping')
                return

            c1 = np.random.choice(p1)
            c2 = np.random.choice(p2)

            p1 = utils.delete_by_value(p1, c1)
            p2 = utils.delete_by_value(p2, c2)

            def insert_randomly(arr, elem):
                n = len(arr)
                return np.insert(arr, np.random.choice(range(n + 1)), elem)

            individual.tours[d_idx1][p_idx1] = insert_randomly(p1, c2)
            individual.tours[d_idx2][p_idx2] = insert_randomly(p2, c1)

        for _ in range(degree):
            id = np.random.choice((0, 1, 2))
            if id == 0:
                reversal_mutation(),
            if id == 1:
                single_customer_rerouting(),
            if id == 2:
                swapping()

    def run_step(self, generation_step):
        '''
        TODO: fix parameter sizes and stuff
        e.g the population size
        '''

        individuals = self.selection()
        new_offspring = []
        p1, p2 = np.random.choice(individuals, 2)
        n_children = int(self.conf['birth_rate'] * self.conf['population_size'])
        intra_depot = False if generation_step % self.conf['extra_depot_every'] == 0 else True

        def step_to_degree(step):
            if step < 5:
                return 10
            if step < 100:
                return 5
            if step < 200:
                return 3
            if step < 500:
                return 2
            return 1
        degree = step_to_degree(generation_step)

        while True:
            offspring = self.create_offspring(p1, p2)
            try:
                self.mutation(offspring, intra_depot=intra_depot, degree=degree)
            except ValueError:
                # could occur if multiple mutations..
                utils.cprint('[r]whoopsie, this individual became a retard :S')
                continue
            if offspring.is_in_feasible_state():
                new_offspring.append(offspring)
            if len(new_offspring) == n_children:
                break

        X = np.concatenate([new_offspring, self.population])
        indices = np.argsort([self.fitness_score(e) for e in X])

        ps = self.conf['population_size']
        best = indices[:ps//2]
        other = np.random.choice(indices[ps//2:], ps//2)
        new_generation = X[np.concatenate((best, other))]

        self.population = new_generation

    def evolve(self, visualize_step=None):
        '''
        repeatedly calls `self.run_step` and eventually plots progres.
        '''
        min_scores = []
        mean_scores = []

        if visualize_step is not None:
            fig, axes = plt.subplots(2, 2)
            axes = axes.flatten()

        for i in range(self.conf['num_generations']):
            self.run_step(i)
            data = [(each, self.fitness_score(each)) for each in self.population]
            fittest, score = min(data, key=lambda x: x[1])
            scores = [each[1] for each in data]
            ms = np.mean(scores)
            min_scores.append(min(scores))
            mean_scores.append(ms)

            if visualize_step is not None and i % visualize_step == 0:
                fittest.visualize(ax=axes[0], title=i)

                axes[1].cla()
                axes[1].plot(min_scores, label='min score')
                axes[1].plot(mean_scores, label='mean score')
                axes[1].legend()
                plt.pause(0.05)
                utils.cprint(f'min score in step [y]{i}[w] is [y]{score}[w] and mean is [y]{ms}')

                axes[2].cla()
                X = [self.fitness_score(e) for e in self.population]
                axes[2].hist(X,bins=100)
                axes[2].set_title('score distribution')

