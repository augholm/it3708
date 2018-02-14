#######################################################################
#                       Ex 1 in bio-inspired
#                       August Holm & Mikael Kvalvaer
#######################################################################
import copy
import loader
import numpy as np  # noqa
from model import MDVRPModel
import utils
from utils import do_profile
import matplotlib.pyplot as plt
import parser as configparser
plt.ion()

filename = 'data/problem/p04'
solution_file = 'data/solution/' + filename.split('/')[2] + '.res'

'''
  puzzle    optimal     achieved    comments
  01:       576         580         700 iterations (595 in 100 iterations)
  02:       473         491         (200 iterations)
  03:       641         673         (300 iterations)
  04:       1001        1092        (1100 after 100 iterations)
  05:       749         801         400 iterations
  06:       876         113         200 iterations
  07:       885
  08:       4437        5362

  801 with 5 10 20 population profile on problem 05. 500 iterations.
  839 with 5 population profile on problem 05. 500 iterations.
  002 with 5 population profile on problem 05. 500 iterations. local optimum..

'''

depots, customers, durations, n_paths_per_depot  = loader.load_dataset(filename)
conf = configparser.parse_config('configs/default.conf')

if len(plt.get_fignums()) > 0:
    ax0, ax1 = plt.gcf().get_axes()
else:
    _, (ax0, ax1) = plt.subplots(1, 2)

model = MDVRPModel(customers, depots, durations, n_paths_per_depot, conf)
optimal_solution = utils.visualize_solution(model, solution_file)
model.evolve(1)

# run me in order to get the best individual
best = min(model.population, key=lambda x: x.fitness_score())

# run me in order to get feasible solution, hopefully
best.minor_modificatios_in_the_end_i_promise()

# run me in order to plot this shit yeee
best.visualize(ax=ax0)
best.describe(disco_mode=True)
