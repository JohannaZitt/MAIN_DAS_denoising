import os

import matplotlib.pyplot as plt
import numpy as np

def extract_id(filename):
    id_part = filename.split('_')[1]
    id_value = id_part.split(':')[1]
    return id_value

def plot_das_data(data):
    channels = data.shape[0]
    alpha=0.7
    i = 0
    plt.figure(figsize=(20, 12))
    for ch in range(channels):
        plt.plot(data[ch][:] + 12 * i, '-k', alpha=alpha)
        i += 1
    plt.show()


'''

Plotting Synthetic Data Seismometer

'''

event_id = 34 # 0, 17, 34, 44
SNR_values = [0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0]


data_path = "data/synthetic_DAS/from_DAS"
event_names_all = os.listdir(data_path)
event_names = [event_name for event_name in event_names_all if str(event_id) == extract_id(event_name)]

#for event_name in event_names:
#    for SNR_value in SNR_values:
#        if SNR_value in

for event_name in event_names:

    print(event_name)

    # Read Data
    data = np.load(os.path.join(data_path, event_name))
    plot_das_data(data)

'''

Plotting Synthetic Data DAS

'''