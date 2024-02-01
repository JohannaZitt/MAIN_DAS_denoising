import os
import re

import matplotlib.pyplot as plt
import numpy as np

def extract_id(filename):
    id = re.search(r'ID:(\d+)', filename).group(1)
    return id


def extract_SNR(filename):
    snr_pattern = re.compile(r'SNR:(\d+(\.\d+)?)\.npy')
    match = re.search(snr_pattern, filename)
    return float(match.group(1))


def plot_das_data(data):
    channels = data.shape[0]
    alpha=0.7
    i = 0
    plt.figure(figsize=(10, 8))
    for ch in range(channels):
        plt.plot(data[ch][:] + 12 * i, '-k', alpha=alpha)
        i += 1
    plt.show()


'''

Plotting Synthetic Data Seismometer

'''

event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 0.3, 1.0, 3.2] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


data_path = "data/synthetic_DAS/from_seis_data"
event_names_all = os.listdir(data_path)
event_names = [event_name for event_name in event_names_all if str(event_id) == extract_id(event_name)]

remove_events = []
for event_name in event_names:
    if not extract_SNR(event_name) in SNR_values:
        remove_events.append(event_name)

for event in remove_events:
    event_names.remove(event)

event_names.sort(key=extract_SNR)
first_event = event_names.pop(0)
event_names.append(first_event)
event_names = event_names[::-1]

# Loading Denoised Data
experiment1 = "07_combined120"
experiment2 = "08_combined480"
denoised_data_path1 = os.path.join("experiments", experiment1, "denoised_synthetic_DAS", "from_seis_data")
denoised_data_path2 = os.path.join("experiments", experiment2, "denoised_synthetic_DAS", "from_seis_data")
denoised_event_names1 = []
denoised_event_names2 = []
for event_name in event_names:
    denoised_event_names1.append("denoised_" + event_name)
for event_name in event_names:
    denoised_event_names2.append("denoised_" + event_name)

t_start = 800
t_end = 1500
ch_start = 30
ch_end = 50
alpha = 0.8
fontsize = 14

# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 3, figsize=(13, 16))

for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end, t_start:t_end]
    denoised_data1 = np.load(os.path.join(denoised_data_path1, denoised_event_names1[i]))[ch_start:ch_end, t_start:t_end]
    denoised_data2 = np.load(os.path.join(denoised_data_path2, denoised_event_names2[i]))[ch_start:ch_end, t_start:t_end]

    # Erste Spalte für noisy Daten:
    n = 0
    for ch in range(data.shape[0]):
        if i == 0:
            axs[i, 0].plot(data[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            axs[i, 0].plot(data[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # Zweite Spalte für denoised_data1
    n = 0
    for ch in range(denoised_data1.shape[0]):
        if i == 0:
            axs[i, 1].plot(denoised_data1[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # Dritte Spalte für denoised_data2
    n = 0
    for ch in range(denoised_data2.shape[0]):
        if i == 0:
            axs[i, 2].plot(denoised_data2[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            axs[i, 2].plot(denoised_data2[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # y-Achsen:
    axs[i, 1].set_yticks([])
    axs[i, 2].set_yticks([])
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fontsize)
    if i == 0:
        axs[i, 0].set_yticks([0, 80, 160, 240, 320], [0, 48, 96, 144, 192], fontsize=fontsize)
    else:
        axs[i, 0].set_yticks([0, 50, 100, 150, 200], [0, 48, 96, 144, 192], fontsize=fontsize)

    # x-Achsen:
    for j in range(3):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 701, 200), [0, 0.5, 1.0, 1.5], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)

# Beschriftung Plot
axs[0, 0].text(30, 350, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(30, 210, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(30, 210, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(30, 210, "SNR: 0.3", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(40, 430, "Synthetic Corrupted Data", fontsize=fontsize)
axs[0, 1].text(60, 430, "Denoised with " + experiment1[3:].capitalize(), fontsize=fontsize)
axs[0, 2].text(60, 430, "Denoised with " + experiment2[3:].capitalize(), fontsize=fontsize)

plt.tight_layout()
plt.savefig("plots/synthetics/Seis_synthetic_denoised1_denoised2.png")
#plt.show()



'''
Plotting Synthetic Data DAS



event_id = 34 # 0, 17, 34, 44
SNR_values = [0.0, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


data_path = "data/synthetic_DAS/from_DAS"
event_names_all = os.listdir(data_path)
event_names = [event_name for event_name in event_names_all if str(event_id) == extract_id(event_name)]

remove_events = []
for event_name in event_names:
    if not extract_SNR(event_name) in SNR_values:
        remove_events.append(event_name)

for event in remove_events:
    event_names.remove(event)

event_names.sort(key=extract_SNR)
first_event = event_names.pop(0)
event_names.append(first_event)
event_names = event_names[::-1]


# Loading Denoised Data
#experiment = "08_combined480"
experiment1 = "07_combined120"
experiment2 = "08_combined480"
denoised_data_path1 = os.path.join("experiments", experiment1, "denoised_synthetic_DAS", "from_DAS")
denoised_data_path2 = os.path.join("experiments", experiment2, "denoised_synthetic_DAS", "from_DAS")
denoised_event_names1 = []
denoised_event_names2 = []
for event_name in event_names:
    denoised_event_names1.append("denoised_" + event_name)
for event_name in event_names:
    denoised_event_names2.append("denoised_" + event_name)

t_start = 800
t_end = 1500
ch_start = 18
ch_end = 62
alpha = 0.8
fontsize = 14

# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 3, figsize=(13, 16))

for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end, t_start:t_end]
    denoised_data1 = np.load(os.path.join(denoised_data_path1, denoised_event_names1[i]))[ch_start:ch_end, t_start:t_end]
    denoised_data2 = np.load(os.path.join(denoised_data_path2, denoised_event_names2[i]))[ch_start:ch_end, t_start:t_end]

    # Erste Spalte für noisy Daten:
    n = 0
    for ch in range(data.shape[0]):
        axs[i, 0].plot(data[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # Zweite Spalte für denoised_data1
    n = 0
    for ch in range(denoised_data1.shape[0]):
        axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1


    # Dritte Spalte für denoised_data2
    n = 0
    for ch in range(denoised_data2.shape[0]):
        axs[i, 2].plot(denoised_data2[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # y-Achsen:
    axs[i, 1].set_yticks([])
    axs[i, 2].set_yticks([])
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fontsize)
    axs[i, 0].set_yticks(range(0, 501, 100), range(0, 9*12*5+1, 9*12), fontsize=fontsize)

    # x-Achsen:
    for j in range(3):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 701, 200), [0, 0.5, 1.0, 1.5], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftung Plot
axs[0, 0].text(30, 470, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(30, 470, "SNR: 10.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(30, 470, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(30, 470, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(40, 560, "Synthetic Corrupted Data", fontsize=fontsize)
axs[0, 1].text(60, 560, "Denoised with " + experiment1[3:].capitalize(), fontsize=fontsize)
axs[0, 2].text(60, 560, "Denoised with " + experiment2[3:].capitalize(), fontsize=fontsize)



plt.tight_layout()
plt.savefig("plots/synthetics/DAS_synthetic_denoised1_denoised2.png")
#plt.show()

'''


