#######################################################################
#                       Ex 1 in bio-inspired
#                       August Holm & Mikael Kvalvaer
#######################################################################

import loader
import numpy as np  # noqa
from model import MDVRPModel
import utils
import matplotlib.pyplot as plt
import parser as configparser
plt.ion()

filename = 'data/problem/p08'
solution_file = 'data/solution/' + filename.split('/')[2] + '.res'

'''

  puzzle    optimal     achieved    comments
  01:       576         580         700 iterations
  02:       473         487         (600 iterations)
  03:       641         673         (300 iterations)
  04:       1001        1092        (1100 after 100 iterations)
  05:       749         905         500 iterations -- stuck at 905 after 500 iterations
  06:       876         113         200 iterations
  07:       885
  08:       4437        5362

'''

depots, customers = loader.load_dataset(filename)
conf = configparser.parse_config('configs/default.conf')

if len(plt.get_fignums()) > 0:
    ax0, ax1 = plt.gcf().get_axes()
else:
    _, (ax0, ax1) = plt.subplots(1, 2)

model = MDVRPModel(customers, depots, conf)
optimal_solution = utils.visualize_solution(model, solution_file)

model.evolve(3)
one = model.population[0]  # debug

L = [each.fitness_score() for each in model.population]
best = model.population[np.argmin(L)]
