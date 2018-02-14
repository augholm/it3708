import itertools
import collections
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

import optimization
import clustering
import utils


class Individual():
    def __init__(self, X, depots, D, N, durations, n_paths_per_depot, initialize=True):
        '''
        X: np array of shape (N+M, 5) -- each row is (x, y, z, a, b) and there are 
            x: x-coordinate
            y: y-coordinate
            z: demand if customer, supply if depot
            a: the depot it belongs to
            b: path index for that depot

        depots: np.array of indices in X that is the depot. shape (M,)

        D: np.array of shape (M+N, M+N). Contains the euclidean distance
        from customer i to customer j in D[i, j]. D[0, j] is the distance
        from depot to customer j.

        '''

        self.X = X
        self.N = N
        self.D = D
        self.depots = depots

        duration_values = list(durations.values())
        if max(duration_values) == min(duration_values) == np.iinfo(np.int64).max:
            self.durations = None
        else:
            self.durations = durations

        self.safe_mode = True

        self.n_customers = self.X.shape[0] - self.depots.shape[0]
        self.n_paths_per_depot = n_paths_per_depot

        '''
        each row in `assignment_info` has the following data:
            0-2: d_idx, p_idx, index in path
            3-4: previous element, next element
            5: capacity of that path

        This data is redundant, and it is used to speed up calculations. It's
        basically used for caching.
        '''
        self.assignment_info = np.ones((self.X.shape[0], 6), np.int64) * -1
        for idx in self.depots:
            self.assignment_info[idx] = np.array([-1, -1, -1, idx, idx, -1])

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
                old_path = np.array(path, copy=True)
                # path = old_path
                path = self.solve_tsp(path)
                L.append(path[1:-1])

            while len(L) < self.n_paths_per_depot:
                L.append(np.array([], dtype=np.int64))
            self.paths[d_idx] = L

        for p, _, _, d_idx, p_idx in self.iter_paths(include_depot=False,
                                                     yield_depot_idx=True,
                                                     yield_path_idx=True):
            self.update_path(p, d_idx, p_idx, check_feasibility_after=self.safe_mode)

    ##############################################################################
    # utility functions
    ##############################################################################

    def update_path(self, path, d_idx, p_idx, check_feasibility_after=False):
        '''

        '''
        if len(path) > 0:
            if path[0] == path[-1] and len(path) > 1:
                pass
            else:
                path = np.hstack((d_idx, path, d_idx))

        cost = self.cost(path)
        cap = self.capacity(path)

        if len(self.paths[d_idx]) < p_idx + 1:
            for _ in range(1 + p_idx - len(self.paths[d_idx])):
                self.paths[d_idx].append(np.array([]))
        old_path = self.paths[d_idx][p_idx]
        self.paths[d_idx][p_idx] = path[1:-1]
        self.costs[(d_idx, p_idx)] = cost
        self.caps[(d_idx, p_idx)] = cap

        for idx, i in enumerate(path[1:-1]):
            self.assignment_info[i] = np.array([d_idx, p_idx, idx, path[idx], path[idx+2], cap])
            # self.depot_assignments[i] = d_idx
            # self.path_assignments[i] = p_idx
            # self.path_indices[i] = idx
            # self.prev[i] = path[idx]
            # self.next[i] = path[idx+2]

        if check_feasibility_after:
            self.is_feasible(1)

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

    def fitness_score(self, demand_penalty=0, duration_penalty=0):
        total_path_cost = sum(list(self.costs.values()))

        if self.durations is not None:
            t_dur = self.total_duration_violation()
        else:
            t_dur = 0
        t_dem = self.total_demand_violation()

        return (total_path_cost +
                t_dem * demand_penalty +
                t_dur * duration_penalty)

    def insert_customer(self, i, d_idx, p_idx, j, check_feasibility_after=False):
        '''
        Assigns after j
        '''
        path = self.paths[d_idx][p_idx]
        idx = np.asscalar(np.argwhere(path == j).flatten())
        new_path = np.hstack((path[:idx+1], i, path[idx+1:]))
        self.update_path(new_path, d_idx, p_idx, check_feasibility_after=check_feasibility_after)

    def remove_customer(self, i, check_feasibility_after=False):
        d_idx, p_idx, idx = self.assignment_info[i, [0, 1, 2]]
        path = self.paths[d_idx][p_idx]
        self.update_path(np.concatenate((path[:idx], path[idx+1:])),
                         d_idx, p_idx, check_feasibility_after=check_feasibility_after)

    def find_customer(self, i):
        '''
        returns the depot index and path index of our customer
        '''
        info = self.assignment_info[i]
        ret_val = info[0], info[1]
        if ret_val[0] == -1 or ret_val[1] == -1:
            return None
        else:
            return ret_val

    # @utils.do_profile
    def RI(self, supply_penalty=1, repair_mode=False, repair_multiplier=1):
        # print('before:', self.fitness_score(0))
        if repair_mode:
            customers = []
            for (d_idx, p_idx), cap in self.caps.items():
                if self.X[d_idx, 2] > cap:
                    customers.append(self.paths[d_idx][p_idx])
            customers = np.unique(np.concatenate(customers))
        else:
            customers = np.arange(self.n_customers)

        for u in customers:
            '''
            One simple way would be to only allow moves if it does not break feasibility
            '''
            run = True
            while run:
                if not repair_mode:
                    run = False
                D = self.D
                N = self.N

                du_idx, pu_idx, u_idx = self.assignment_info[u, [0, 1, 2]]

                V = N[u,:]
                Q = self.assignment_info[V,:2]
                ## L = []
                ## for each in Q:
                ##     L.append(self.caps[tuple(each)])
                u_path_demand = self.caps[du_idx, pu_idx]


                '''
                STEP 1 --- set up the neighbouring variables and demands.

                u_path:    .... a, u, b, c, ....
                v_path:    .... f, v, g, h, ...

                self.assignment_info: (depot_index, path_index, index in the path (ex. depot), previous element, nxt element)
                '''
                # u_path_with_depots = np.hstack((du_idx, self.paths[du_idx][pu_idx], du_idx))
                u_path_with_depots = np.concatenate([[du_idx], self.paths[du_idx][pu_idx], [du_idx]])
                a, u, b = u_path_with_depots[u_idx:u_idx+3]

                X = self.assignment_info[V, :]
                F, V, G, H = X[:, 3], V, X[:, 4], self.assignment_info[X[:, 4], 4]
                c = self.assignment_info[b, 4]
                L = []
                # for d_idx, p_idx in self.assignment_info[V, :2]:
                #     L.append(self.caps[d_idx, p_idx])
                # V_path_demand = np.array(L)  # checked: seem correct
                V_path_demand = self.assignment_info[V, 5]

                u_demand, b_demand = self.X[u, 2], self.X[b, 2]
                u_path_supply = self.X[du_idx, 2]

                V_demand, G_demand = self.X[V, 2], self.X[G, 2]
                V_path_supply = self.X[X[:, 0], 2]

                same_path_mask = np.all(X[:, [0, 1]] == np.array([du_idx, pu_idx]), axis=1)

                '''
                STEP 2 --- find the change in violation if action is done.
                '''
                zeros = np.zeros_like(V_path_demand)
                Q_V = np.vstack((
                    zeros + u_demand,
                    zeros + u_demand + b_demand,
                    zeros + u_demand + b_demand,
                    zeros - V_demand + u_demand,
                    zeros - V_demand + u_demand + b_demand,
                    zeros - V_demand - G_demand + u_demand + b_demand))
                # Q_V = np.array(Q_v, copy=True)

                Q_U = np.vstack((
                    zeros - u_demand,
                    zeros - u_demand - b_demand,
                    zeros - u_demand - b_demand,
                    zeros - u_demand + V_demand,
                    zeros - u_demand - b_demand + V_demand,
                    zeros - u_demand - b_demand + V_demand + G_demand))
                # Q_U = np.array(Q_u, copy=True)

                V_violations = V_path_demand - V_path_supply
                u_violation = u_path_demand - u_path_supply

                u_violations_before_capped = np.clip(np.zeros_like(Q_U) + u_violation, 0, 1000000)
                u_violations_after_capped = np.clip((Q_U + u_violation), 0, 100000)

                V_violations_before_capped = np.clip(np.zeros_like(Q_V) + V_violations, 0, 100000)
                V_violations_after_capped = np.clip((Q_V + V_violations), 0, 100000)

                reduced_V_violations = V_violations_before_capped - V_violations_after_capped
                reduced_u_violations = u_violations_before_capped - u_violations_after_capped
                reduced_violations = reduced_V_violations + reduced_u_violations
                # import ipdb; ipdb.set_trace()


                '''
                STEP 3 --- find the delta costs for each action
                '''
                remove_u = -D[a,u] - D[u, b]
                remove_v = -D[F,V] - D[V,G]
                remove_ub = -D[a,u] - D[b,c]
                remove_vg = -D[F,V] - D[G,H]

                # this variable explains how the path cost will be changed
                # for different actions.
                deltas = np.ones((6, V.shape[0])) * np.inf

                delta = remove_u + D[a,b] + D[V,u] + D[u,G]
                deltas[0, ~same_path_mask] = delta[~same_path_mask]

                delta = remove_u + D[a,V] + D[V,b] + remove_v + D[F,u] + D[u,G]
                deltas[3, ~same_path_mask] = delta[~same_path_mask]

                if b not in self.depots:  # then I can do M2, M3 M5
                    delta = remove_ub + D[a,c] - D[V,G] + D[V,u] + D[b,G]
                    deltas[1, ~same_path_mask] = delta[~same_path_mask]
                    delta = -D[a,u] - D[b,c] + D[a,c] - D[V,G] + D[V,b] + D[u,G]
                    deltas[2, ~same_path_mask] = delta[~same_path_mask]

                    mask = ~np.isin(H, self.depots) & ~same_path_mask
                    delta = remove_ub + remove_vg + D[a,V] + D[G,c] + D[F,u] + D[b,H] 
                    delta = remove_ub + D[a,V] + D[G,c] - D[F,V] - D[G,H] + D[F,u] + D[b,H]
                    deltas[5, mask] = delta[mask]

                '''
                STEP 4 -- Find the best action
                '''

                pdeltas = np.array(deltas, copy=True)  # just a debug variable


                scores = np.zeros_like(deltas)

                if repair_mode:
                    if self.total_demand_violation() == 0:
                        return
                    scores = deltas - reduced_violations * repair_multiplier
                else:
                    penalty = np.clip(-reduced_violations, 0, 10000)
                    scores = deltas + penalty * supply_penalty

                if self.durations is not None:
                    V_durations = np.array([self.durations[d_idx] for d_idx in self.assignment_info[V, 0]], np.int64)
                    V_costs = []
                    for d_idx, p_idx in self.assignment_info[V, :2]:
                        V_costs.append(self.costs[d_idx,p_idx])
                    V_costs = np.array(V_costs, np.int64)
                    mask = (V_costs <= V_durations) & ((V_costs + deltas) > V_durations)
                    duration_penalty = mask.astype(int)

                    mask = (V_costs > V_durations) & ((V_costs + deltas) <= V_durations)
                    duration_bonus = mask.astype(int)
                    if duration_penalty.any():
                        scores += duration_penalty * scores[scores != np.inf].mean()
                    # if duration_bonus.any():
                    #     scores -= duration_bonus * 100

                action, idx = np.unravel_index(scores.argmin(), scores.shape)
                # print(self, self.total_duration_violation(), scores[action,idx])
                if scores[action, idx] >= -1e-08:
                    run = False
                    break
                # print(f'u is {u}. expected delta: {pdeltas[action,idx]}. Expected reduction in feas.: {reduced_violations[action,idx]}. action/idx {action,idx}')

                '''
                STEP 5 --- update the paths
                '''
                v, g = V[idx], G[idx]
                du_idx, pu_idx, u_idx = self.assignment_info[u, [0, 1, 2]]
                dv_idx, pv_idx, v_idx = self.assignment_info[v, [0, 1, 2]]

                u_path = self.paths[du_idx][pu_idx]
                v_path = self.paths[dv_idx][pv_idx]

                u_path = np.hstack((du_idx, u_path, du_idx))
                v_path = np.hstack((dv_idx, v_path, dv_idx))
                u_idx += 1
                v_idx += 1

                # now find a way to include feasibility in the score :)
                if action == 0:
                    new_u_path = np.hstack((u_path[:u_idx], u_path[u_idx+1:]))
                    new_v_path = np.hstack((v_path[:v_idx+1], u, v_path[v_idx+1:]))
                elif action == 1:
                    new_u_path = np.hstack((u_path[:u_idx], u_path[u_idx+2:]))
                    new_v_path = np.hstack((v_path[:v_idx+1], u, b, v_path[v_idx+1:]))
                elif action == 2:
                    new_u_path = np.hstack((u_path[:u_idx], u_path[u_idx+2:]))
                    new_v_path = np.hstack((v_path[:v_idx+1], b, u, v_path[v_idx+1:]))
                elif action == 3:
                    new_u_path = np.hstack((u_path[:u_idx], v, u_path[u_idx+1:]))
                    new_v_path = np.hstack((v_path[:v_idx], u, v_path[v_idx+1:]))
                elif action == 5:
                    new_u_path = np.hstack((u_path[:u_idx], v, g, u_path[u_idx+2:]))
                    new_v_path = np.hstack((v_path[:v_idx], u, b, v_path[v_idx+2:]))
                if action == 4:
                    print('got a 4 action. wtf?')
                    break


                def info():
                    print(f'{u_path} -> {new_u_path}')
                    print(f'{v_path} -> {new_v_path}')
                    print(action, u, v)
                if not repair_mode:
                    before = self.total_demand_violation()
                    sb = self.fitness_score(0)
                self.is_feasible(1)

                self.update_path(new_u_path, du_idx, pu_idx, check_feasibility_after=False)
                self.update_path(new_v_path, dv_idx, pv_idx, check_feasibility_after=self.safe_mode)
                if not repair_mode:
                    after = self.total_demand_violation()
                    sa = self.fitness_score(0)

                    # print(f'self: {self}. violation: {before} -> {after} (expected {reduced_violations[action,idx]}). Score: {sb:.4} -> {sa:.4}. (Expected {deltas[action,idx]:.4}). Action idx is {action,idx} and u is {u}')

                '''
                DONE
                '''


    def route_improvement(self, penalty=1):
        '''
        path includes the depot!

        u_path:    .... a, u, b, c, ....
        v_path:    .... f, v, g, h, ...
        '''
        # customers = np.setdiff1d(np.arange(self.X.shape[0]), self.depots)
        customers = np.arange(50)
        D = self.D
        ccat = np.hstack

        for u in customers:
            du_idx, pu_idx = self.find_customer(u)
            u_path = np.hstack((du_idx, self.paths[du_idx][pu_idx], du_idx))
            u_idx = self.assignment_info[u, 2] + 1
            a, b = u_path[u_idx-1], u_path[u_idx+1]
            options = [1, 4]
            if u_idx <= len(u_path) - 3:
                options += [2, 3, 5]
            options = np.random.permutation(options).tolist()

            done = False
            for v in self.N[u]:
                if done: break

                dv_idx, pv_idx = self.find_customer(v)
                if du_idx == dv_idx and pu_idx == pv_idx:
                    continue

                # this cannot be removed because we often update the stuff
                v_path = np.hstack((dv_idx, self.paths[dv_idx][pv_idx], dv_idx))

                # BEWARE: if depot, we have to increase the idx
                v_idx = self.assignment_info[v, 2] + 1

                if 2 in options and v_idx + 2 < len(v_path):
                    options += [6]
                else:
                    options = utils.delete_by_value(options, 6).tolist()

                # TODO: what if this is not possible?
                f, g = v_path[v_idx-1], v_path[v_idx+1]
                if 2 in options:
                    c = u_path[u_idx+2]
                if 6 in options:
                    h = v_path[v_idx+2]

                delta = np.inf
                for opt in options:
                    if opt == 1:
                        # (M1)
                        #         ... a, b, ...                ... f, v, u, g, ...
                        delta = -D[a,u] - D[u,b] + D[a,b] - D[v,g] + D[v,u] + D[u,g]
                        if delta > 0:
                            continue
                        new_path_u = ccat((u_path[:u_idx], u_path[u_idx+1:]))
                        new_path_v = ccat((v_path[:v_idx+1], u, v_path[v_idx+1:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2
                    elif opt == 2:
                        delta = -D[a,u] - D[b,c] + D[a,c] - D[v,g] + D[v,u] + D[b,g]
                        if delta > 0:
                            continue
                        new_path_u = ccat((u_path[:u_idx], u_path[u_idx+2:]))
                        new_path_v = ccat((v_path[:v_idx+1], u, b, v_path[v_idx+1:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2
                    elif opt == 3:
                        delta = -D[a,u] - D[b,c] + D[a,c] -D[v,g] + D[b,v] + D[u,g]
                        if delta > 0:
                            continue
                        new_path_u = ccat((u_path[:u_idx], u_path[u_idx+2:]))
                        new_path_v = ccat((v_path[:v_idx+1], b, u, v_path[v_idx+1:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2
                    elif opt == 4:
                        delta = -D[a,u] - D[u,b] + D[a,v] + D[v,b] - D[f,v] - D[v,g] + D[f,u] + D[u,g]
                        if delta > 0:
                            continue
                        new_path_u = ccat((u_path[:u_idx], v, u_path[u_idx+1:]))
                        new_path_v = ccat((v_path[:v_idx], u, v_path[v_idx+1:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2
                    elif opt == 5:
                        delta = -D[a,u] - D[b,c] + D[a,v] + D[v,c] - D[f,v] - D[v,g] + D[f,u] + D[b,g]
                        if delta > 0:
                            continue
                        new_path_u = ccat((u_path[:u_idx], v, u_path[u_idx+2:]))
                        new_path_v = ccat((v_path[:v_idx], u, b, v_path[v_idx+1:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2
                    elif opt == 6:
                        delta = -D[a,u] - D[b,c] + D[a,v] + D[g,c] - D[f,v] - D[g,h] + D[f,u] + D[b,h]
                        if delta >= 0:
                            continue
                        new_path_u = ccat((u_path[:u_idx], v, g, u_path[u_idx+2:]))
                        new_path_v = ccat((v_path[:v_idx], u, b, v_path[v_idx+2:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2

                    elif opt == 8:
                        delta = -D[u,b] - D[b,c] + D[u,v] + D[v,c] - D[f,v] - D[v,g] + D[f,b] + D[b,g]
                        if delta >= 0: continue
                        new_path_u = ccat((u_path[:u_idx+1], v, u_path[u_idx+2:]))
                        new_path_v = ccat((v_path[:v_idx], u_patch[u_idx+1], v_path[v_idx+1:]))
                        pen1 = 0 if self.capacity(new_path_u) <= 80 else self.costs[(du_idx, pu_idx)] * penalty
                        pen2 = 0 if self.capacity(new_path_v) <= 80 else self.costs[(dv_idx, pv_idx)] * penalty
                        delta = delta + pen1 + pen2


                    if delta < 0:
                        def info():
                            print(u_path, '->', new_path_u)
                            print(v_path, '->', new_path_v)
                            print(u,v, opt)
                        self.update_path(new_path_u, du_idx, pu_idx, check_feasibility_after=False)
                        self.update_path(new_path_v, dv_idx, pv_idx, check_feasibility_after=self.safe_mode)
                        done = True
                        break


    def total_demand_violation(self):
        total = 0
        for (d_idx, p_idx), demand in self.caps.items():
            supply = self.X[d_idx, 2]
            total += max(0, demand - supply)
        return total

    def total_duration_violation(self):
        total = 0
        for (d_idx, p_idx), cost in self.costs.items():
            total += int(cost > self.durations[d_idx])
        return total

    def repair(self):
        if self.average_capacity_infeasibility() == 0:
            return
        self.RI(supply_penalty=1, repair_mode=True, repair_multiplier=5)
        if self.total_demand_violation() == 0:
            return
        self.RI(supply_penalty=10, repair_mode=True, repair_multiplier=10)
        if self.total_demand_violation() == 0:
            self.RI(supply_penalty=1000, repair_mode=True, repair_multiplier=10)
        if self.average_capacity_infeasibility() == 0:
            pass

    def repair_duration(self):
        '''
        ad-hoc function that tries to repair duration for paths
        '''

        self.is_feasible()
        D = self.D

        for d_idx in self.depots:
            free_paths = [i for i, each in enumerate(self.paths[d_idx]) if len(each) == 0]
            costs = np.array([self.costs[d_idx, p_idx]
                              for p_idx in range(self.n_paths_per_depot) ], np.int64)
            if costs.max() <= self.durations[d_idx]:
                continue
            if free_paths:
                p_idx = np.argmax(costs)
                path = self.paths[d_idx][p_idx]
                new_path_1 = path[:len(path)//2]
                new_path_2 = path[len(path)//2:]
                def wr(p):
                    return self.cost(np.hstack((d_idx, p, d_idx)))
                # print(free_paths, self.durations[d_idx], wr(path), wr(new_path_1), wr(new_path_2))
                # import ipdb; ipdb.set_trace()
                self.update_path(new_path_1, d_idx, p_idx, check_feasibility_after=False)
                self.update_path(new_path_2, d_idx, free_paths[0], check_feasibility_after=True)

        for p, cost, _, d_idx, p_idx in self.iter_paths(yield_depot_idx=True, yield_path_idx=True):
            if self.durations[d_idx] < cost:
                excess = cost - self.durations[d_idx]

                while excess >= 0:
                    old = excess
                    excess = self.costs[d_idx,p_idx] - self.durations[d_idx]
                    if excess <= 0:
                        break

                    # print(d_idx, p_idx, excess)
                    path = self.paths[d_idx][p_idx]
                    path_with_depot = np.hstack((d_idx, path, d_idx))
                    pwd = path_with_depot
                    delta = -D[pwd[:-2],path] - D[path,pwd[2:]] + D[pwd[:-2],pwd[2:]]

                    # True in mask indicates that we will get rid of infeasibility for this route
                    mask = delta + excess < 0
                    if not np.any(mask):
                        idx = np.argmin(delta)
                        customer = path[idx]
                    else:
                        idx = np.random.choice(np.argwhere(mask).flatten())
                        customer = path[idx]
                    d_cands = self.depots[self.depots != d_idx]
                    def wr(p):
                        return self.cost(np.hstack((d_idx, p, d_idx)))
                    self.move_customer(customer, repair_mode=True)


    def minor_modificatios_in_the_end_i_promise(self):
        while True:
            violations = self.total_duration_violation(), self.total_demand_violation()
            print(violations)
            if violations == (0, 0):
                return
            self.repair()
            self.repair_duration()
            if np.random.choice((0, 1), p=(0.9, 0.1)):
                self.RI()

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

    def optimally_insert_customer(self, i, depot_id=None, not_in=None, repair_mode=False):
        '''
        depot_id: int or list of ints
        '''

        # that way, if depot_id is int, we now treat it as an array.
        if depot_id is not None:
            depot_id = np.array(depot_id, np.int64)

        demand = self.X[i, 2]
        # just find a random path initially, but try to find closest path for i
        _, d_idx, p_idx = utils.choose_random_paths(self, depot_id=depot_id, include_indices=True, min_length=0)[0]
        candidate_paths = []
        candidate_paths.append((d_idx,p_idx))
        for k in range(self.N.shape[1]):
            neighbour_customer = self.N[i, k]
            x = self.find_customer(neighbour_customer)
            if x is not None:
                if not_in is not None and x in not_in:
                    continue
                if repair_mode and (self.caps[x] + demand > self.X[d_idx, 2]):
                    continue
                if depot_id is None:
                    d_idx, p_idx = x
                    candidate_paths.append((d_idx, p_idx))
                elif depot_id is not None and x[0] in depot_id:
                    d_idx, p_idx = x
                    candidate_paths.append((d_idx, p_idx))

        candidate_paths = np.array(candidate_paths)
        old_ = np.array(candidate_paths, copy=True)
        # we want to preserve the order of `candidate_path` after taking out
        # the unique elements
        _, indices = np.unique(candidate_paths, axis=0, return_index=True)
        candidate_paths = candidate_paths[indices]
        D = self.D
        L = []
        for d_idx, p_idx in candidate_paths:
            P = np.hstack((d_idx, self.paths[d_idx][p_idx], d_idx))
            delta = D[P[:-1], i] + D[i, P[1:]] - D[P[:-1], P[1:]]
            idx = np.argmin(delta)
            delta_cost = delta[idx]
            score = delta_cost
            if self.durations is not None:
                if self.caps[d_idx,p_idx] + self.X[i,2] <= self.X[d_idx, 2]:
                    score -= 500
                if (self.costs[d_idx,p_idx] < self.durations[d_idx]) and ((self.costs[d_idx,p_idx] + delta_cost) > self.durations[d_idx]):
                    score += 1000
                if (self.costs[d_idx,p_idx] < self.durations[d_idx]) and ((self.costs[d_idx,p_idx] + delta_cost) <= self.durations[d_idx]):
                    score -= 1000

            new_path = np.hstack((P[:idx+1], i, P[idx+1:]))
            L.append((d_idx, p_idx, new_path, score))

        d_idx, p_idx, new_path, score = min(L, key=lambda x: x[3])
            # TODO: theres some bug here
        if len(new_path) > 3 and len(np.unique(new_path[1:-1])) != len(new_path[1:-1]):
            import ipdb; ipdb.set_trace()
        bef = self.costs[d_idx,p_idx]
        self.update_path(new_path, d_idx, p_idx)
        aft=self.costs[d_idx,p_idx]
        return

    def delete_customer(self, i, return_indices=False):
        d_idx, p_idx = self.find_customer(i)
        path_with_customer = self.paths[d_idx][p_idx]
        path_without_customer = utils.delete_by_value(self.paths[d_idx][p_idx], i)
        self.update_path(path_without_customer, d_idx, p_idx, check_feasibility_after=False)
        if return_indices:
            return d_idx, p_idx


    def move_customer(self, i, depot_id=None, strategy='best', repair_mode=False):
        '''
        i: the index of the customer
        strategy: 'best'|'random'
        depot_id: int or list
        '''

        d_idx, p_idx = self.delete_customer(i, return_indices=True)
        # self.optimally_insert_customer(i, depot_id=depot_id)
        self.optimally_insert_customer(i, not_in=[(d_idx, p_idx)], repair_mode=repair_mode)

        # d_idx, p_idx = self.find_customer(i)
        # self.insert_customer(d_idx, p_idx)
        # path_with_customer = self.paths[d_idx][p_idx]
        # path_without_customer = utils.delete_by_value(self.paths[d_idx][p_idx], i)
        # self.insert_customer(i, d_idx, p_idx)


    def average_capacity_infeasibility(self):
        L = []
        for k, v in self.caps.items():
            depot_idx, _ = k
            capacity = self.X[depot_idx, 2]
            L.append(v > capacity)
        return np.mean(L)
                
    def is_feasible(self, v=False):
        L = []
        for each in self.paths.values():
            for path in each:
                for c in path:
                    L.append(c)
        L = np.array(L)
        L = sorted(L)

        n_entries = self.X[:, 0].shape[0]
        n_customers = n_entries - len(self.depots)
        if not len(L) == n_customers:
            U = np.unique(L)
            L = np.array(L)
            D = np.setdiff1d(np.arange(n_customers), U)
            import ipdb; ipdb.set_trace()
            return False
        if len(L) != len(np.unique(L)):
            U = np.unique(L)
            L = np.array(L)
            D = np.setdiff1d(L, U)
            import ipdb; ipdb.set_trace()
            return False
            pass
        return True

    def solve_tsp(self, path):
        if path[0] != path[-1] or len(path) <= 1:
            raise Exception('wrong type path')

        indices = optimization.solve_tsp(self.X[path[:-1], 0:2], circular_indices=False, start_index=0)
        new_path = path[indices]
        if not np.all(np.sort(path) == np.sort(new_path)):
            import ipdb; ipdb.set_trace()
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
        mask = np.ones_like(self.X[:, 0], np.bool)
        mask[self.depots] = False
        customers = self.X[mask]

        ax.scatter(depots[:, 0], depots[:, 1], c='r', marker='*')
        ax.scatter(customers[:, 0], customers[:, 1], s=2*customers[:, 2], c='b', marker='.')
        # for txt, x, y in zip(self.customers[:, 0], self.customers[:, 1], self.customers[:, 2]):
        #     txt = str(txt)
        #     ax.annotate(txt, (x, y), size=8)

        for color, depot_id in zip('c m y k r g b'.split(), self.depots):
            for path, _, _ in self.iter_paths(depot_id=depot_id, include_depot=True):
                xy_coords = self.X[path][:, 0:2]
                ax.plot(xy_coords[:, 0], xy_coords[:, 1], color, alpha=0.6)
        if title is not None:
            ax.set_title(title)
        pass

    def describe(self, disco_mode=False):
        print(self.fitness_score())
        L = []
        for p, c, cap, d_idx, p_idx in self.iter_paths(yield_depot_idx=True, yield_path_idx=True):
            L.append((p, c, cap, d_idx, p_idx))
        L = sorted(L, key=lambda x: (x[3], x[4]))

        for p, c, cap, d_idx, p_idx in L:
            if c == 0:  # not a path
                continue
            d_idx = d_idx - self.depots[0] + 1
            p_idx = p_idx + 1
            
            p = ' '.join([f'{e:3}' for e in np.hstack((0, p, 0))])
            print_args = [f'{d_idx:3}', f'{p_idx:3}', f'{float(c):10.5}', p]
            if disco_mode:
                for i, each in enumerate(print_args):
                    c = np.random.choice(('r g b c m y w').split())
                    print_args[i] = f'[{c}]{each}'
                utils.cprint(' '.join(print_args))
            else:
                print(*print_args)
