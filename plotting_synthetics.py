import os
import re
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def extract_id(filename):
    id = re.search(r"ID:(\d+)", filename).group(1)
    return id


def extract_SNR(filename):
    snr_pattern = re.compile(r"SNR:(\d+(\.\d+)?)\.npy")
    match = re.search(snr_pattern, filename)
    return float(match.group(1))


def plot_das_data(data):
    channels = data.shape[0]
    alpha=0.7
    i = 0
    plt.figure(figsize=(10, 8))
    for ch in range(channels):
        plt.plot(data[ch][:] + 15 * i, "-k", alpha=alpha)
        i += 1

    plt.show()

def xcorr(x, y):  # Code by Martijn van den Ende

    # FFT of x and conjugation
    X_bar = np.fft.rfft(x).conj()
    Y = np.fft.rfft(y)

    # Compute norm of data
    norm_x_sq = np.sum(x ** 2)
    norm_y_sq = np.sum(y ** 2)
    norm = np.sqrt(norm_x_sq * norm_y_sq)

    # Correlation coefficients
    R = np.fft.irfft(X_bar * Y) / norm

    # Return correlation coefficient
    return np.max(R)


def compute_xcorr_window(x):  # Code by Martijn van den Ende
    Nch = x.shape[0]
    Cxy = np.zeros((Nch, Nch)) * np.nan

    for i in range(Nch):
        for j in range(i):
            Cxy[i, j] = xcorr(x[i], x[j])

    return np.nanmean(Cxy)


def compute_moving_coherence(data, bin_size):  # Code by Martijn van den Ende

    N_ch = data.shape[0]

    cc = np.zeros(N_ch)

    for i in range(N_ch):
        start = max(0, i - bin_size // 2)
        stop = min(i + bin_size // 2, N_ch)
        ch_slice = slice(start, stop)
        cc[i] = compute_xcorr_window(data[ch_slice])

    return cc


"""
Estemate Velocity of waveform

data_path = "data/synthetic_DAS/from_DAS/cleanDAS_ID:34_SNR:0.npy"
data = np.load(os.path.join(data_path))
#data = data[47:55,1080:1115]

experiment = "03_accumulation_horizontal"

print(data.shape)
plot_das_data(data)
"""

""" imshow - synthetically corrupted DAS data  """

# Parameters:
cmap = "plasma"
vmin = -1.5
vmax = 1.5
ch_start = 10
ch_end = 70
ch_total=60
fs=16
t_start_wiggle = 450
t_end_wiggle=650
channel_wiggle_comparison=32#32

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

experiment = "03_accumulation_horizontal"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_DAS")
denoised_event_names = []

for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 4, figsize=(12, 14), gridspec_kw={"width_ratios": [5, 5, 1, 5]})

ground_truth_data = np.load(os.path.join(data_path, "cleanDAS_ID:34_SNR:0.npy"))[ch_start:ch_end]
ground_truth_data = ground_truth_data[:, 550:1750]
for i, event_name in enumerate(event_names):
    # Load Data:
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end]
    data = data[:, 550:1750]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[ch_start:ch_end]
    denoised_data = denoised_data[:, 550:1750]

    # Calculate CC
    bin_size = 11
    raw_cc = compute_moving_coherence(data, bin_size)
    denoised_cc = compute_moving_coherence(denoised_data, bin_size)
    raw_denoised_cc = denoised_cc / raw_cc
    raw_denoised_cc = raw_denoised_cc[ch_start:ch_end]
    raw_denoised_cc = raw_denoised_cc[::-1]
    x = np.arange(raw_denoised_cc.shape[0])
    y_seis = raw_denoised_cc
    X_seis = np.vstack((x, y_seis)).T
    X_seis = np.vstack((X_seis[:, 1], X_seis[:, 0])).T

    # Plotting Data:
    axs[i, 0].imshow(data, cmap=cmap, aspect="auto", interpolation="antialiased",
          vmin=vmin, vmax=vmax)
    axs[i, 1].imshow(denoised_data, cmap=cmap, aspect="auto", interpolation="antialiased",
                     vmin=vmin, vmax=vmax)
    axs[i, 2].plot(X_seis[:, 0], X_seis[:, 1], color = "black")
    axs[i, 2].axvline(x=1, color="black", linestyle="dotted")

    # Plotting Wiggle Comparison
    axs[i, 3].plot(ground_truth_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="red",
                   label="Ground Truth Data", linewidth=1.5, alpha=0.5, zorder=1)
    if not i == 0:
        axs[i, 3].plot(data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="grey",
                       label="Noise Corrupted Data",
                       linewidth=2, alpha=0.25, zorder=1)
    axs[i, 3].plot(denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="black",
                   label="Denoised Data", linewidth=1.5, alpha=1, zorder=1)

    # Legend
    axs[i, 3].legend(fontsize=12)

    # Label and Ticks
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fs)
    axs[i, 0].set_yticks([59, 49, 39, 29, 19, 9], [0.0, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=fs-2)
    axs[i, 1].set_yticks([59, 49, 39, 29, 19, 9], [0.0, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=fs-2)
    axs[i, 1].set_yticklabels([])
    axs[i, 2].set_yticks([])
    axs[i, 2].set_xlim(0, 6)
    axs[i, 2].set_ylim(0, raw_denoised_cc.shape[0]-1)
    axs[i, 3].set_yticks([])
    axs[i, 3].set_xticks([50, 100, 150], [0.2, 0.3, 0.4], fontsize=fs-2)

    axs[i, 0].set_xticks([200, 400, 600, 800, 1000], [0.5, 1.0, 1.5, 2.0, 2.5], fontsize=fs-2)
    axs[i, 1].set_xticks([200, 400, 600, 800, 1000], [0.5, 1.0, 1.5, 2.0, 2.5], fontsize=fs-2)
    axs[i, 2].set_xticks([1, 3, 5])
    if i == 3:
        axs[i, 0].set_xlabel("Time [s]", fontsize=fs)
        axs[i, 1].set_xlabel("Time [s]", fontsize=fs)
        axs[i, 2].set_xlabel("Gain [-]", fontsize=fs)
        axs[i, 3].set_xlabel("Time [s]", fontsize=fs)
    else:
        axs[i, 0].set_xticklabels([])
        axs[i, 1].set_xticklabels([])
        axs[i, 2].set_xticklabels([])
        axs[i, 3].set_xticklabels([])

    ax2 = axs[i, 3].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("Strain Rate [norm.]", fontsize=fs - 2)

    # plot arrows
    arrow_style = "fancy,head_width=0.5,head_length=0.5"
    axs[i, 0].annotate("", xy=(0, ch_total - channel_wiggle_comparison),
                       xytext=(-0.05, ch_total - channel_wiggle_comparison),
                       arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))


axs[3, 0].annotate("", xy=(t_start_wiggle, 59.5),
                        xytext=(t_start_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_start_wiggle, 59.5),
                        xytext=(t_start_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 0].annotate("", xy=(t_end_wiggle, 59.5),
                        xytext=(t_end_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_end_wiggle, 59.5),
                        xytext=(t_end_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))



axs[0, 0].set_title("Noise Corrupted", fontsize=fs+4, y=1.05)
axs[0, 1].set_title("Denoised Data", fontsize=fs+4, y=1.05)
axs[0, 2].set_title("LWC", fontsize=fs+4, y=1.05)
axs[0, 3].set_title("Wiggle Comparison", fontsize=fs+4, y=1.05)
axs[3, 2].set_xticks([1, 3, 5], [1, 3, 5], fontsize=fs-2)
#axs[3, 0].set_xticks([0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0], fontsize=fs-2)


# add patch:
#rect = patches.Rectangle((500, 28), 100, 1, linewidth=1.5, edgecolor="black", facecolor="none")
#axs[0, 0].add_patch(rect)

# Add letters in plots:
letter_params = {
        "fontsize": fs+2,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
    }
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p", "q"]
snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0"]

for i in range(4):
    axs[i, 0].text(x=0.09, y=1, transform=axs[i, 0].transAxes, s=snr_values[i], **letter_params)
    for j in range(4):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*4 + j], **letter_params)



plt.tight_layout()
plt.savefig("plots/synthetics/waterfall+wiggle_comparison_DAS_legend.pdf", dpi=400)
#plt.show()






""" imshow - synthetic DAS data 

# Parameters:
cmap = "plasma"
vmin = -1.5
vmax = 1.5
ch_start = 10
ch_end = 70
ch_total = 60
fs=16

t_start_wiggle = 1050
t_end_wiggle = 1250
channel_wiggle_comparison = 28#29

event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0

data_path = "data/synthetic_DAS/from_seis"
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

experiment = "03_accumulation_horizontal"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_seis")
denoised_event_names = []

for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


fig, axs = plt.subplots(len(event_names), 4, figsize=(12, 14), gridspec_kw={"width_ratios": [5, 5, 1, 5]})

ground_truth_data = np.load(os.path.join(data_path, "clean_ID:46_SNR:0.npy"))[ch_start:ch_end]

for i, event_name in enumerate(event_names):
    # Load Data:
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[ch_start:ch_end]

    # Calculate CC
    bin_size = 11
    raw_cc = compute_moving_coherence(data, bin_size)
    denoised_cc = compute_moving_coherence(denoised_data, bin_size)
    raw_denoised_cc = denoised_cc / raw_cc
    raw_denoised_cc = raw_denoised_cc[ch_start:ch_end]
    raw_denoised_cc = raw_denoised_cc[::-1]
    x = np.arange(raw_denoised_cc.shape[0])
    y_seis = raw_denoised_cc
    X_seis = np.vstack((x, y_seis)).T
    X_seis = np.vstack((X_seis[:, 1], X_seis[:, 0])).T

    # Plotting Data:
    axs[i, 0].imshow(data, cmap=cmap, aspect="auto", interpolation="antialiased",
          vmin=vmin, vmax=vmax)
    axs[i, 1].imshow(denoised_data, cmap=cmap, aspect="auto", interpolation="antialiased",
                     vmin=vmin, vmax=vmax)
    axs[i, 2].plot(X_seis[:, 0], X_seis[:, 1], color = "black")
    axs[i, 2].axvline(x=1, color="black", linestyle="dotted")

    # Plotting wiggle for wiggle comparison
    axs[i, 3].plot(ground_truth_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="red",
                       label="Ground Truth Data", linewidth=1.5, alpha=0.5, zorder=1)
    if not i == 0:
        axs[i, 3].plot(data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="grey", label="Noise Corrupted Data",
                       linewidth=2, alpha=0.25, zorder=1)
    axs[i, 3].plot(denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="black",
                   label="Denoised Data", linewidth=1.5, alpha=1, zorder=1)

    # legend
    # axs[i, 3].legend(fontsize = 20)



    # Label and Ticks
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fs)
    axs[i, 0].set_yticks([59, 49 ,39, 29, 19, 9, 0], [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize = fs-4)
    axs[i, 1].set_yticks([59, 49 ,39, 29, 19, 9, 0], [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=fs - 4)
    axs[i, 1].set_yticklabels([])
    axs[i, 2].set_yticks([])
    axs[i, 2].set_xlim(0, 9)
    axs[i, 2].set_ylim(0, raw_denoised_cc.shape[0]-1)
    axs[i, 3].set_yticks([])

    axs[i, 0].set_xticks([400, 800, 1200, 1600, 2000], [1.0, 2.0, 3.0, 4.0, 5.0], fontsize = fs-4)
    axs[i, 1].set_xticks([400, 800, 1200, 1600, 2000], [1.0, 2.0, 3.0, 4.0, 5.0], fontsize = fs-4)
    axs[i, 2].set_xticks([1, 7], [1, 7], fontsize = fs-4)
    axs[i, 3].set_xticks([50, 100, 150], [0.2, 0.3, 0.4], fontsize = fs-4)
    if i == 3:
        axs[i, 0].set_xlabel("Time [s]", fontsize=fs)
        axs[i, 1].set_xlabel("Time [s]", fontsize=fs)
        axs[i, 2].set_xlabel("Gain [-]", fontsize=fs)
        axs[i, 3].set_xlabel("Time [s]", fontsize=fs)
    else:
        axs[i, 0].set_xticklabels([])
        axs[i, 1].set_xticklabels([])
        axs[i, 2].set_xticklabels([])
        axs[i, 3].set_xticklabels([])

    ax2 = axs[i, 3].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("Ground Velocity [norm.]", fontsize=fs-2)

    # plot arrows
    arrow_style = "fancy,head_width=0.5,head_length=0.5"
    axs[i, 0].annotate("", xy=(0, ch_total - channel_wiggle_comparison),
                       xytext=(-0.05, ch_total - channel_wiggle_comparison),
                       arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=2))


axs[3, 0].annotate("", xy=(t_start_wiggle, 59.5),
                        xytext=(t_start_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_start_wiggle, 59.5),
                        xytext=(t_start_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 0].annotate("", xy=(t_end_wiggle, 59.5),
                        xytext=(t_end_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))
axs[3, 1].annotate("", xy=(t_end_wiggle, 59.5),
                        xytext=(t_end_wiggle, 59.9),
                        arrowprops=dict(color="black", arrowstyle=arrow_style, linewidth=1))


axs[0, 0].set_title("Noise Corrupted", fontsize=fs+4, y=1.05)
axs[0, 1].set_title("Denoised Data", fontsize=fs+4, y=1.05)
axs[0, 2].set_title("LWC", fontsize=fs+4, y=1.05)
axs[0, 3].set_title("Wiggle Comparison", fontsize=fs+4, y=1.05)


# add patch:
# rect = patches.Rectangle((1000, 10), 250, 1, linewidth=1, edgecolor="black", facecolor="none")
# axs[0, 0].add_patch(rect)

# Add letters in plots:
letter_params = {
        "fontsize": fs+2,
        "verticalalignment": "top",
        "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
    }
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0"]

for i in range(4):
    axs[i, 0].text(x=0.09, y=1, transform=axs[i, 0].transAxes, s=snr_values[i], **letter_params)
    for j in range(4):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*4 + j], **letter_params)



plt.tight_layout()
#plt.savefig("plots/synthetics/water_fall_wiggle_seis_synthetic.pdf", dpi=400)
plt.show()



"""



'''

Single Waveform Comparison - Seis



event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


data_path = "data/synthetic_DAS/from_seis"
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
experiment = "03_accumulation_horizontal"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_seis")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 1050
t_end = 1250
channel = 28#29
alpha = 0.8
fontsize = 14


fig, axs = plt.subplots(1, len(event_names), figsize=(18, 4))
for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[channel, t_start:t_end]

    # Plot Data
    axs[i].plot(data, color="red", alpha=0.5, linewidth=1, label="Noise Corrupted")
    axs[i].plot(denoised_data, color="black", linewidth=1, label="Denoised")

    # Achses
    axs[i].set_ylim([-22, 22])
    axs[i].set_ylim([-22, 22])
    axs[i].set_yticks([])
    if i == 0:
        axs[i].set_ylabel("Strain Rate [norm]", fontsize=fontsize)
    if i == 3:
        axs[i].legend(fontsize=fontsize) #, loc="lower left"




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
#plt.savefig("plots/synthetics/seis_wiggle_comparison.png", dpi=400)
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
experiment = "03_accumulation_horizontal"
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_DAS")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 950
t_end = 1250
channel = 32#32
alpha = 0.8
fontsize = 14


fig, axs = plt.subplots(1, len(event_names), figsize=(18, 4))
for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[channel, t_start:t_end]
    denoised_data = np.load(os.path.join(denoised_data_path, denoised_event_names[i]))[channel, t_start:t_end]

    # Plot Data
    axs[i].plot(data, color="red", alpha=0.5, linewidth=1, label="Noise Corrupted")
    axs[i].plot(denoised_data, color="black", linewidth=1, label="Denoised")

    # Achses
    axs[i].set_ylim([-12, 12])
    axs[i].set_ylim([-12, 12])
    axs[i].set_yticks([])
    if i == 0:
        axs[i].set_ylabel("Strain Rate [norm]", fontsize=fontsize)
    if i == 3:
        axs[i].legend(fontsize=fontsize)




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
#plt.savefig("plots/synthetics/das_wiggle_comparison.png", dpi=400)
plt.show()
'''


'''

Single Waveforms DAS




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
#plt.savefig("plots/synthetics/das_single_waveform.png")
plt.show()
'''


'''

Single Waveform Comparison Seis



event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


data_path = "data/synthetic_DAS/from_seis"
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
denoised_data_path = os.path.join("experiments", experiment, "denoised_synthetic_DAS", "from_seis")
denoised_event_names = []
for event_name in event_names:
    denoised_event_names.append("denoised_" + event_name)


t_start = 1050
t_end = 1250
channel = 28
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
#plt.savefig("plots/synthetics/seis_single_waveform.png")
plt.show()

'''


'''

Plotting Synthetic Data Seismometer - Section Plots - Two Denoised Versions



event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 0.3, 1.0, 3.2] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


data_path = "data/synthetic_DAS/from_seis"
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
experiment1 = "03_accumulation_horizontal"
experiment2 = "06_combined800"
denoised_data_path1 = os.path.join("experiments", experiment1, "denoised_synthetic_DAS", "from_seis")
denoised_data_path2 = os.path.join("experiments", experiment2, "denoised_synthetic_DAS", "from_seis")
denoised_event_names1 = []
denoised_event_names2 = []
for event_name in event_names:
    denoised_event_names1.append("denoised_" + event_name)
for event_name in event_names:
    denoised_event_names2.append("denoised_" + event_name)

t_start = 800
t_end = 1500
ch_start = 20
ch_end = ch_start+15
ch_black = 28 - ch_start
alpha = 0.5
fontsize = 13

# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 3, figsize=(11, 14))

for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end, t_start:t_end]
    denoised_data1 = np.load(os.path.join(denoised_data_path1, denoised_event_names1[i]))[ch_start:ch_end, t_start:t_end]
    denoised_data2 = np.load(os.path.join(denoised_data_path2, denoised_event_names2[i]))[ch_start:ch_end, t_start:t_end]

    # Erste Spalte für noisy Daten:
    n = 0
    for ch in range(data.shape[0]):
        print(ch)
        if i == 0:
            if ch == ch_black:
                axs[i, 0].plot(data[ch][:] + 20 * n, '-k')
            else:
                axs[i, 0].plot(data[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            if ch == ch_black:
                axs[i, 0].plot(data[ch][:] + 12 * n, '-k')
            else:
                axs[i, 0].plot(data[ch][:] + 12 * n, '-k', alpha=alpha)

        n += 1

    # Zweite Spalte für denoised_data1
    n = 0
    for ch in range(denoised_data1.shape[0]):
        if i == 0:
            if ch == ch_black:
                axs[i, 1].plot(denoised_data1[ch][:] + 20 * n, '-k')
            else:
                axs[i, 1].plot(denoised_data1[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            if ch == ch_black:
                axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k')
            else:
                axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # Dritte Spalte für denoised_data2
    n = 0
    for ch in range(denoised_data2.shape[0]):
        if i == 0:
            if ch == ch_black:
                axs[i, 2].plot(denoised_data2[ch][:] + 20 * n, '-k')
            else:
                axs[i, 2].plot(denoised_data2[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            if ch == ch_black:
                axs[i, 2].plot(denoised_data2[ch][:] + 12 * n, '-k')
            else:
                axs[i, 2].plot(denoised_data2[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # y-Achsen:
    axs[i, 1].set_yticks([])
    axs[i, 2].set_yticks([])
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fontsize)
    if i == 0:
        axs[i, 0].set_yticks([0, 80, 160, 240], [0, 48, 96, 144], fontsize=fontsize)
    else:
        axs[i, 0].set_yticks([0, 50, 100, 150], [0, 48, 96, 144], fontsize=fontsize)

    # x-Achsen:
    for j in range(3):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 701, 200), [0, 0.5, 1.0, 1.5], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)

# Beschriftung Plot
axs[0, 0].text(30, 240, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(30, 150, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(30, 150, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(30, 150, "SNR: 0.3", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(40, 330, "Synthetic Corrupted Data", fontsize=fontsize)
axs[0, 1].text(60, 320, "Denoised with Acc_Horiz", fontsize=fontsize) #+ experiment1[3:].capitalize()
axs[0, 2].text(45, 320, "Denoised with Combined 800", fontsize=fontsize) #+ experiment2[3:].capitalize()

plt.tight_layout()
#plt.savefig("plots/synthetics/Seis_synthetic_denoised1_denoised2.png")
plt.show()
'''

'''

Plotting Synthetic Data Seismometer - Section Plots - One Denoised Versions



event_id = 46 # 37, 46, 48, 96
SNR_values = [0.0, 1.0, 3.2, 10.0] # 0.0, 0.3, 1.0, 3.2, 10.0, 31.6, 100.0


data_path = "data/synthetic_DAS/from_seis"
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
experiment1 = "03_accumulation_horizontal"
denoised_data_path1 = os.path.join("experiments", experiment1, "denoised_synthetic_DAS", "from_seis")
denoised_event_names1 = []

for event_name in event_names:
    denoised_event_names1.append("denoised_" + event_name)


t_start = 1030
t_end = 1230
ch_start = 20
ch_end = ch_start+15
ch_black = 28 - ch_start
alpha = 0.5
fontsize = 13

# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 2, figsize=(9, 14))

for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end, t_start:t_end]
    denoised_data1 = np.load(os.path.join(denoised_data_path1, denoised_event_names1[i]))[ch_start:ch_end, t_start:t_end]

    # Erste Spalte für noisy Daten:
    n = 0
    for ch in range(data.shape[0]):
        print(ch)
        if i == 0:
            if ch == ch_black:
                axs[i, 0].plot(data[ch][:] + 20 * n, '-k')
            else:
                axs[i, 0].plot(data[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            if ch == ch_black:
                axs[i, 0].plot(data[ch][:] + 12 * n, '-k')
            else:
                axs[i, 0].plot(data[ch][:] + 12 * n, '-k', alpha=alpha)

        n += 1

    # Zweite Spalte für denoised_data1
    n = 0
    for ch in range(denoised_data1.shape[0]):
        if i == 0:
            if ch == ch_black:
                axs[i, 1].plot(denoised_data1[ch][:] + 20 * n, '-k')
            else:
                axs[i, 1].plot(denoised_data1[ch][:] + 20 * n, '-k', alpha=alpha)
        else:
            if ch == ch_black:
                axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k')
            else:
                axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', alpha=alpha)
        n += 1

    # y-Achsen:
    axs[i, 1].set_yticks([])
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fontsize)
    if i == 0:
        axs[i, 0].set_yticks([0, 80, 160, 240], [0, 48, 96, 144], fontsize=fontsize)
    else:
        axs[i, 0].set_yticks([0, 50, 100, 150], [0, 48, 96, 144], fontsize=fontsize)

    # x-Achsen:
    for j in range(2):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 201, 100), [0, 0.25, 0.5], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)

# Beschriftung Plot
x_dist = 7
axs[0, 0].text(x_dist, 300, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(x_dist, 177, "SNR: 10.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(x_dist, 171, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(x_dist, 171, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(55, 340, "Noise Corrupted", fontsize=fontsize+2)
axs[0, 1].text(80, 330, "Denoised", fontsize=fontsize+2) #+ experiment1[3:].capitalize()


# Beschriftungen
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

for i in range(4):
    for j in range(2):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*3+j], **letter_params)




plt.tight_layout()
#plt.savefig("plots/synthetics/seis_synthetic_denoised1.png", dpi=400)
plt.show()
'''

'''
Plotting Synthetic Data DAS - Section Plots - One Denoised Version



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
experiment1 = "03_accumulation_horizontal"

denoised_data_path1 = os.path.join("experiments", experiment1, "denoised_synthetic_DAS", "from_DAS")
denoised_event_names1 = []
for event_name in event_names:
    denoised_event_names1.append("denoised_" + event_name)

t_start = 950
t_end = 1250
ch_start = 24
ch_end = 55
alpha = 0.5
fontsize = 13
linewidth=0.8
ch_black = 26

# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 2, figsize=(9, 14))

for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end, t_start:t_end]
    denoised_data1 = np.load(os.path.join(denoised_data_path1, denoised_event_names1[i]))[ch_start:ch_end, t_start:t_end]

    # Erste Spalte für noisy Daten:
    n = 0
    for ch in range(data.shape[0]):
        if ch == ch_black:
            axs[i, 0].plot(data[ch][:] + 12 * n, '-k', linewidth=linewidth)
        else:
            axs[i, 0].plot(data[ch][:] + 12 * n, '-k', linewidth=linewidth, alpha=alpha)
        n += 1

    # Zweite Spalte für denoised_data1
    n = 0
    for ch in range(denoised_data1.shape[0]):
        if ch == ch_black:
            axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', linewidth=linewidth)
        else:
            axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', linewidth=linewidth, alpha=alpha)
        n += 1


    # y-Achsen:
    axs[i, 1].set_yticks([])
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fontsize)
    axs[i, 0].set_yticks(range(0, 301, 100), [0, 64, 128, 192], fontsize=fontsize)

    # x-Achsen:
    for j in range(2):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 301, 100), [0, 0.25, 0.5, 0.75], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftung Plot
x = 14
y = 364
axs[0, 0].text(x, y+3, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(x, y, "SNR: 10.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(x, y, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(x, y, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(70, 410, "Noise Corrupted", fontsize=fontsize+2)
axs[0, 1].text(90, 410, "Denoised", fontsize=fontsize+2) #+ experiment1[3:].capitalize()

# Beschriftungen
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l", "m", "n", "o", "p"]
letter_params = {
    "fontsize": fontsize,
    "verticalalignment": "top",
    "bbox": {"edgecolor": "k", "linewidth": 1, "facecolor": "w",}
}

for i in range(4):
    for j in range(2):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*3+j], **letter_params)



plt.tight_layout()
plt.savefig("plots/synthetics/DAS_synthetic_denoised1.png", dpi=400)
#plt.show()

'''


'''
Plotting Synthetic Data DAS - Section Plots - Two Denoised Versions



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
experiment1 = "01_ablation_horizontal"
experiment2 = "06_combined800"
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
alpha = 0.5
fontsize = 13
linewidth=0.8
ch_black = 32

# Erstellen von Subplots mit Größe (15, 12)
fig, axs = plt.subplots(len(event_names), 3, figsize=(11, 14))

for i, event_name in enumerate(event_names):
    # Laden der Daten
    data = np.load(os.path.join(data_path, event_name))[ch_start:ch_end, t_start:t_end]
    denoised_data1 = np.load(os.path.join(denoised_data_path1, denoised_event_names1[i]))[ch_start:ch_end, t_start:t_end]
    denoised_data2 = np.load(os.path.join(denoised_data_path2, denoised_event_names2[i]))[ch_start:ch_end, t_start:t_end]

    # Erste Spalte für noisy Daten:
    n = 0
    for ch in range(data.shape[0]):
        if ch == ch_black:
            axs[i, 0].plot(data[ch][:] + 12 * n, '-k', linewidth=linewidth)
        else:
            axs[i, 0].plot(data[ch][:] + 12 * n, '-k', linewidth=linewidth, alpha=alpha)
        n += 1

    # Zweite Spalte für denoised_data1
    n = 0
    for ch in range(denoised_data1.shape[0]):
        if ch == ch_black:
            axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', linewidth=linewidth)
        else:
            axs[i, 1].plot(denoised_data1[ch][:] + 12 * n, '-k', linewidth=linewidth, alpha=alpha)
        n += 1


    # Dritte Spalte für denoised_data2
    n = 0
    for ch in range(denoised_data2.shape[0]):
        if ch == ch_black:
            axs[i, 2].plot(denoised_data2[ch][:] + 12 * n, '-k', linewidth=linewidth)
        else:
            axs[i, 2].plot(denoised_data2[ch][:] + 12 * n, '-k', linewidth=linewidth, alpha=alpha)
        n += 1

    # y-Achsen:
    axs[i, 1].set_yticks([])
    axs[i, 2].set_yticks([])
    axs[i, 0].set_ylabel("Offset [m]", fontsize=fontsize)
    axs[i, 0].set_yticks(range(0, 501, 100), [0, 64, 128, 192, 256, 320], fontsize=fontsize)

    # x-Achsen:
    for j in range(3):
        axs[0, j].set_xticks([])
        axs[1, j].set_xticks([])
        axs[2, j].set_xticks([])
        axs[3, j].set_xticks(range(0, 701, 200), [0, 0.5, 1.0, 1.5], fontsize=fontsize)
        axs[3, j].set_xlabel("Time [s]", fontsize=fontsize)


# Beschriftung Plot
x = 30
y = 495
axs[0, 0].text(x, y, "No Noise Added", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[1, 0].text(x, y, "SNR: 10.0", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[2, 0].text(x, y, "SNR: 3.2", bbox=dict(facecolor='white'), fontsize=fontsize)
axs[3, 0].text(x, y, "SNR: 1.0", bbox=dict(facecolor='white'), fontsize=fontsize)

axs[0, 0].text(40, 560, "Synthetic Corrupted Data", fontsize=fontsize)
axs[0, 1].text(60, 560, "Denoised with Acc_Horiz", fontsize=fontsize) #+ experiment1[3:].capitalize()
axs[0, 2].text(60, 560, "Denoised with Combined800", fontsize=fontsize) #+ experiment2[3:].capitalize()

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





