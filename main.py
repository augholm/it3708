#######################################################################
#                       Ex 1 in bio-inspired
#                       August Holm & Mikael Kvalvaer
#######################################################################

import sample
import loader
import numpy as np  # noqa
from model import MDVRPModel
import utils
import matplotlib.pyplot as plt
import parser
plt.ion()

filename = 'data/problem/p01'
# 01: 576 (610 best)
# 02: 473 (502 best)
# 03: 641 (746 best, 10%)
# 04: 1001 (1238)
solution_file = 'data/solution/' + filename.split('/')[2] + '.res'
with open(solution_file) as f:
    optimal_score = float(f.readline().strip())

depots, customers = loader.load_dataset(filename)

conf = parser.parse_config('configs/default.conf')
# conf['lala']


model = MDVRPModel(customers, depots, conf)

model.evolve(3)
one = model.population[0]

# L = [each.fitness_score() for each in model.population]
L = [each.fitness_score() for each in model.population]
best = model.population[np.argmin(L)]

if len(plt.get_fignums()) > 0:
    ax0, ax1 = plt.gcf().get_axes()

best.visualize(ax=ax0)

import copy
clone = copy.deepcopy(np.random.choice(model.population))
print(clone.total_violation())
clone.repair()
print(clone.total_violation())
