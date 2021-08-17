import os
import matplotlib.pyplot as plt
from numpy import mean, std

NAMES = []
AVERAGES = []
STDS = []
AVG_DISTANCES = []
DISTANCE_STDS = []




for file in os.listdir('saved_weights'):
    if file.endswith('.txt'):
        cur_file = open('saved_weights/' + file)
        NAMES.append(int(file.split('_')[-1].split('.')[0]))
        my_list = cur_file.readlines()
        AVERAGES.append(float(my_list[0].split()[6]))
        TOTAL_COSTS = [float(thing.strip()) for thing in my_list[2:22]]
        STDS.append(std(TOTAL_COSTS))
        l1_norm = [float(thing.strip()) for thing in my_list[24:44]]
        AVG_DISTANCES.append(mean(l1_norm))
        DISTANCE_STDS.append(std(l1_norm))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(15,15))

ax1.scatter(NAMES, AVERAGES)
ax1.title.set_text('Average Cost above Heuristic Cost (lower is better)')

ax2.scatter(NAMES, STDS)
ax2.title.set_text('Standard Deviation of Cost above Heuristic Cost')

ax3.scatter(NAMES, AVG_DISTANCES)
ax3.title.set_text('Average Distance from Goals (0 means success)')

ax4.scatter(NAMES, DISTANCE_STDS)
ax4.title.set_text('Standard Deviation of Distance from Goals')

fig.savefig('Results.png')
os.startfile('Results.png')
