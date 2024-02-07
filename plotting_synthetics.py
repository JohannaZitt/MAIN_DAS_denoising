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
        plt.plot(data[ch][:] + 15 * i, '-k', alpha=alpha)
        i += 1

    plt.show()


'''
Estemate Velocity of waveform

'''

data_path = "data/synthetic_DAS/from_DAS/cleanDAS_ID:34_SNR:0.npy"
data = np.load(os.path.join(data_path))
data = data[47:55,1080:1115]

print(data.shape)
plot_das_data(data)



'''

Single Waveform Comparison - Seis



event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 0.3, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


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
experiment = "08_combined480"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_seis_data")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 1050
t_end = 1250
channel = 30
alpha = 0.8
fontsize = 12


fig, axs = plt.subplots(1, len(event_names), figsize=(18, 4))
for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[channel, t_start:t_end]

    # Plot Data
    axs[i].plot(data, color="black", linewidth=1, label="Noise Coruppted")
    axs[i].plot(denoised_data, color="red", linewidth=1, label="Denoised")

    # Achses
    axs[i].set_ylim([-22, 22])
    axs[i].set_ylim([-22, 22])
    axs[i].set_yticks([])
    if i == 0:
        axs[i].set_ylabel("Strain Rate [norm]", fontsize=fontsize)
        axs[i].legend(fontsize=fontsize-2, loc="lower left")




    axs[i].set_xticks([0, 100, 200], [0, 0.25, 0.5], fontsize=fontsize)
    axs[i].set_xlabel("Time [s]", fontsize=fontsize)


    # Beschriftungen
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0", "SNR: 0.3"]
    letter_params = {
        "fontsize": fontsize,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
    }

    axs[i].text(x=0.0, y=1.0, transform=axs[i].transAxes, s=letters[i], **letter_params)
    axs[i].text(x=0.5, y=1.03, transform=axs[i].transAxes, s=snr_values[i], fontsize=fontsize + 2, ha="center")

plt.tight_layout()
#plt.savefig("plots/synthetics/seis_wiggle_comparison.png")
plt.show()

'''

'''

Single Waveform Comparison - DAS



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
experiment = "08_combined480"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_DAS")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 950
t_end = 1250
channel = 32#32
alpha = 0.8
fontsize = 12


fig, axs = plt.subplots(1, len(event_names), figsize=(14, 4))
for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[channel, t_start:t_end]

    # Plot Data
    axs[i].plot(data, color="black", linewidth=1, label="Noise Coruppted")
    axs[i].plot(denoised_data, color="red", linewidth=1, label="Denoised")

    # Achses
    axs[i].set_ylim([-12, 12])
    axs[i].set_ylim([-12, 12])
    axs[i].set_yticks([])
    if i == 0:
        axs[i].set_ylabel("Strain Rate [norm]", fontsize=fontsize)
        axs[i].legend(fontsize=fontsize-2)




    axs[i].set_xticks([0, 100, 200, 300], [0, 0.25, 0.5, 0.75], fontsize=fontsize)
    axs[i].set_xlabel("Time [s]", fontsize=fontsize)


    # Beschriftungen
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0"]
    letter_params = {
        "fontsize": fontsize,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
    }

    axs[i].text(x=0.0, y=1.0, transform=axs[i].transAxes, s=letters[i], **letter_params)
    axs[i].text(x=0.5, y=1.03, transform=axs[i].transAxes, s=snr_values[i], fontsize=fontsize + 2, ha="center")

plt.tight_layout()
plt.savefig("plots/synthetics/das_wiggle_comparison.png")
plt.show()
'''


'''

Single Waveform Comparison DAS



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
experiment = "08_combined480"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_DAS")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 950
t_end = 1250
channel = 32#32
alpha = 0.8
fontsize = 12


fig, axs = plt.subplots(2, len(event_names), figsize=(16, 6))
for j, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[j]))[channel, t_start:t_end]

    # Plot Data
    axs[0, j].plot(data, color="black", linewidth=1)
    axs[1, j].plot(denoised_data, color="black", linewidth=1)

    # Achses
    axs[0, j].set_ylim([-12, 12])
    axs[1, j].set_ylim([-12, 12])
    if j == 0:
        axs[0, j].set_ylabel("Noise Corrupted", fontsize=fontsize+2)
        axs[1, j].set_ylabel("Denoised", fontsize=fontsize+2)
    axs[0, j].set_yticks([])
    axs[1, j].set_yticks([])

    axs[0, j].set_xticks([])
    axs[1, j].set_xticks([0, 100, 200, 300], [0, 0.25, 0.5, 0.75], fontsize=fontsize)
    axs[1, j].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftungen
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

for j in range(len(event_names)):
    axs[0, j].text(x=0.5, y=1.03, transform=axs[0, j].transAxes, s=snr_values[j], fontsize=fontsize+2, ha="center")
    for i in range(2):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*len(event_names)+j], **letter_params)

plt.tight_layout()
plt.savefig("plots/synthetics/das_single_waveform.png")
#plt.show()
'''


'''

Single Waveform Comparison Seis



event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 0.3, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


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
experiment = "08_combined480"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_seis_data")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 1050
t_end = 1250
channel = 30
alpha = 0.8
fontsize = 12


fig, axs = plt.subplots(2, len(event_names), figsize=(16, 6))
for j, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[j]))[channel, t_start:t_end]

    # Plot Data
    axs[0, j].plot(data, color="black", linewidth=1)
    axs[1, j].plot(denoised_data, color="black", linewidth=1)

    # Achses
    axs[0, j].set_ylim([-24, 24])
    axs[1, j].set_ylim([-24, 24])
    if j == 0:
        axs[0, j].set_ylabel("Noise Corrupted", fontsize=fontsize+2)
        axs[1, j].set_ylabel("Denoised", fontsize=fontsize+2)
    axs[0, j].set_yticks([])
    axs[1, j].set_yticks([])

    
    #putting y labelto the right side
    #if j == len(event_names)-1:
    #    print('in if')
        #ax0 = axs[0, j].twinx()
        #ax1 = axs[1, j].twinx()
        #ax0.set_ylabel("Strain Rate [norm]", fontsize=fontsize, rotation=-90, labelpad=15)
        #ax1.set_ylabel("Strain Rate [norm]", fontsize=fontsize, rotation=-90, labelpad=15)
        #ax0.set_yticks([])
        #ax1.set_yticks([])
        #axs[0, j].set_ylabel("Strain Rate [norm]", fontsize=fontsize, rotation=-90, position=(1, 1))
        #axs[0, j].yaxis.tick_right()
        #axs[1, j].yaxis.tick_right()
    #else:
    #    axs[0, j].set_yticks([])
    #    axs[1, j].set_yticks([])
    

    axs[0, j].set_xticks([])
    axs[1, j].set_xticks([0, 100, 200], [0, 0.25, 0.5], fontsize=fontsize)
    axs[1, j].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftungen
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0", "SNR: 0.3"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

for j in range(len(event_names)):
    axs[0, j].text(x=0.5, y=1.03, transform=axs[0, j].transAxes, s=snr_values[j], fontsize=fontsize+2, ha="center")
    for i in range(2):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*len(event_names)+j], **letter_params)

plt.tight_layout()
plt.savefig("plots/synthetics/seis_single_waveform.png")
#plt.show()

'''


'''

Plotting Synthetic Data Seismometer - Section Plots



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

'''
Plotting Synthetic Data DAS - Section Plots



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
ch_start = 20
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
    axs[i, 0].set_yticks(range(0, 501, 100), range(0, 8*12*5+1, 8*12), fontsize=fontsize)

    # x-Achsen:
    for j in range(3):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 701, 200), [0, 0.5, 1.0, 1.5], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftung Plot
x = 30
y = 498
axs[0, 0].text(x, y, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(x, y, "SNR: 10.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(x, y, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(x, y, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(40, 560, "Synthetic Corrupted Data", fontsize=fontsize)
axs[0, 1].text(60, 560, "Denoised with " + experiment1[3:].capitalize(), fontsize=fontsize)
axs[0, 2].text(60, 560, "Denoised with " + experiment2[3:].capitalize(), fontsize=fontsize)

# Beschriftungen
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

for i in range(4):
    for j in range(3):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*3+j], **letter_params)



plt.tight_layout()
#plt.savefig("plots/synthetics/DAS_synthetic_denoised1_denoised2.png")
plt.show()

'''






