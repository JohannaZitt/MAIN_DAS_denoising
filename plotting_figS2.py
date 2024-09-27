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

col_pink = "#CE4A75"

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
    axs[i, 3].plot(ground_truth_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color=col_pink,
                   label="Raw", linewidth=1.5, alpha=0.8, zorder=1)
    if not i == 0:
        axs[i, 3].plot(data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="grey",
                       label="Synthetics",
                       linewidth=1.5, alpha=0.6, zorder=1)
    axs[i, 3].plot(denoised_data[channel_wiggle_comparison][t_start_wiggle:t_end_wiggle], color="black",
                   label="Denoised", linewidth=1.5, alpha=0.8, zorder=1)


    # Legend
    #axs[i, 3].legend(fontsize=15)

    # Label and Ticks
    axs[i, 0].set_ylabel("Offset [km]", fontsize=fs)
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
        axs[i, 2].set_xlabel("Gain []", fontsize=fs)
        axs[i, 3].set_xlabel("Time [s]", fontsize=fs)
    else:
        axs[i, 0].set_xticklabels([])
        axs[i, 1].set_xticklabels([])
        axs[i, 2].set_xticklabels([])
        axs[i, 3].set_xticklabels([])

    ax2 = axs[i, 3].twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("Amplitude [norm.]", fontsize=fs)

    # plot arrows
    arrow_style = "fancy,head_width=0.5,head_length=1.8"
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



axs[0, 0].set_title("Synthetics", fontsize=fs+4, y=1.05)
axs[0, 1].set_title("Denoised", fontsize=fs+4, y=1.05)
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
letters = ["a", "b", "c", "d", "e", "f ", "g", "h", "i ", "j ", "k", "l ", "m", "n", "o", "p", "q"]
snr_values = ["No Noise Added", "SNR: 10", "SNR: 3.2", "SNR: 1.0"]

for i in range(4):
    axs[i, 0].text(x=0.12, y=1, transform=axs[i, 0].transAxes, s=snr_values[i], **letter_params)
    for j in range(4):
        axs[i, j].text(x=0.0, y=1.0, transform=axs[i, j].transAxes, s=letters[i*4 + j], **letter_params)



plt.tight_layout()
#plt.savefig("plots/synthetics/waterfall+wiggle_comparison_DAS_synthetics.pdf", dpi=400)
plt.show()