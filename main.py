import loader
import numpy as np  # noqa
import utils
import sample
from model import MDVRPModel
import matplotlib.pyplot as plt
plt.ion()

filename = 'data/problem/p01'
depots, customers = loader.load_dataset(filename)

model = MDVRPModel(customers, depots, generate_initial_population=True)

plt.gcf().close()
model.evolve(visualize_step=1)
plt.clf()
