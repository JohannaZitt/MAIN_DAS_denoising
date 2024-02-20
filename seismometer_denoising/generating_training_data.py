import numpy as np
import math
import random

'''


Create geometry of receivers on Rhonegletscher for receiver RA81-RA88

Die source_location wird zufällig gewählt während des trainings x-coordinate im range von 672000 + i*50 (i in range(11))
und die y-coordinate von 160950 + j*50 (j in range(15)) and z-coordinate in range von 2450 - l*50 (l in range(4))



'''

def compute_distance(point1, point2):
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)
    return distance



# receiver and source coordinates
receiver_names = ["RA81", "RA82", "RA83", "RA84", "RA85", "RA86", "RA87", "RA88"]
receiver_positions = [[672254, 161419, 2514], [672438, 161365, 2497], [672412, 161139, 2475], [672279, 161057, 2463],
                      [672113, 161145, 2480], [672103, 161365, 2510], [672262, 161252, 2488], [672250, 161572, 2537]]
initial_source_location = [672000, 160950, 2450]
fs = 400

# choose source location randomly on grid space. synthetic locations are 50 m apart:
x = [50 * random.randint(0,11), 50 * random.randint(0,14), - 50 * random.randint(0, 4)]
src_loc = np.array(initial_source_location) + np.array(x)

# compute distance from source to receiver position:
distances = []
for receiver_position in receiver_positions:
    distance = compute_distance(src_loc, receiver_position)
    distances.append(distance)

# choose velocity randomly
vel_max = 3900.
vel_min = 1650.
velocity = random.randint(1650, 3900)
slowness = 1/velocity

shifts = []
for distance in distances:
    shift = distance * slowness * fs
    shifts.append(int(shift))

print(shifts)



# compute distance between source location and receiver:




