#######################################################################
#                       Ex 1 in bio-inspired
#                       August Holm & Mikael Kvalvaer
#######################################################################

import loader
import numpy as np  # noqa
from model import MDVRPModel
import utils
import matplotlib.pyplot as plt
import parser
plt.ion()

filename = 'data/problem/p01'
# 01: 576 (610 best)
# 02: 473 (526 best, 11%)
# 03: 641 (746 best, 16%)
# 04: 1001 (1238)
solution_file = 'data/solution/' + filename.split('/')[2] + '.res'
with open(solution_file) as f:
    optimal_score = float(f.readline().strip())

depots, customers = loader.load_dataset(filename)

conf = parser.parse_config('configs/default.conf')
# conf['lala']


model = MDVRPModel(customers, depots, conf)

one = model.population[0]

L = [each.fitness_score() for each in model.population]
print('before')
print(L)
model.evolve(10)
L = [each.fitness_score() for each in model.population]
print('after')
print(L)
# model.evolve(visualize_step=50)
# # 
# scores = [model.fitness_score(e) for e in model.population]
# best = model.population[np.argmin(scores)]
# # best.describe()
# # 
# min_score = min(scores)
# above = '{:.2f}'.format((min_score / optimal_score - 1)*100)
# min_score = '{:.2f}'.format(min_score)
# 
# utils.cprint(f'[g]Done.[w] Min score is [b]{min_score}[w], which is [b] {above}%[w] above optimal') # noqa



# puzzle 1: 633 / 576 ~ 1.09

# with the lookup thingy
# --> 21.347 seconds
# 11.920 in path_cost-
# the roll-function took 5.347 seconds

# 20.856 seconds for the lookup table
# copy.deepcopy took 8 seconds
# 16 seconds with the TSP stuff
# 16 seconds without the TSP heuristic
