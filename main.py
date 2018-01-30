#######################################################################
#                       Ex 1 in bio-inspired
#                       August Holm & Mikael Kvalvaer
#######################################################################

import loader
import numpy as np  # noqa
from model import MDVRPModel
import matplotlib.pyplot as plt
import parser
plt.ion()

filename = 'data/problem/p01'
depots, customers = loader.load_dataset(filename)

conf = parser.parse_config('configs/default.conf')
# conf['lala']

model = MDVRPModel(customers, depots, conf)
#model.evolve(visualize_step=10)
