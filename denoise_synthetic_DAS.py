import os
import keras
import numpy as np
import matplotlib.pyplot as plt


def denoise_file (DAS_data, model, timesamples = 1024, N_sub = 11):

    # split data in 1024 data point sections
    n_ch, n_t = DAS_data.shape
    n_samples = int(n_t / timesamples) + 1
    data = np.zeros((n_samples, n_ch, timesamples))
    for i in range(n_samples - 1):
        data[i] = DAS_data[:, i * timesamples: (i + 1) * timesamples]
    data[n_samples - 1] = DAS_data[:, n_t - timesamples: n_t]

    DAS_reconstructions = np.zeros_like(data)
    N_samples, N_ch, N_t = data.shape

    # loop over every batche of 1024 sampling points
    for n, eval_sample in enumerate(data):

        # prepare sample and masks
        masks = np.ones((N_ch, N_sub, N_t, 1))
        eval_samples = np.zeros_like(masks)

        gutter = N_sub // 2
        mid = N_sub // 2

        for i in range(gutter):
            masks[i, i] = 0
            eval_samples[i, :, :, 0] = eval_sample[:N_sub]

        for i in range(gutter, N_ch - gutter):
            start = i - mid
            stop = i + mid + 1

            masks[i, mid] = 0
            eval_samples[i, :, :, 0] = eval_sample[start:stop]

        for i in range(N_ch - gutter, N_ch):
            masks[i, i - N_ch] = 0
            eval_samples[i, :, :, 0] = eval_sample[-N_sub:]

        # set mean = 0
        for i in range(eval_samples.shape[0]): # for every channel
            for j in range(eval_samples.shape[1]): # for every N_sub
                eval_samples[i, j, :] = eval_samples[i, j, :] - np.mean(eval_samples[i, j, :])

        # Create J-invariant reconstructions
        results = model.predict((eval_samples, masks))
        DAS_reconstructions[n] = np.sum(results, axis=1)[:, :, 0]

    # reshape data into original shape
    data_save = np.zeros_like(DAS_data)
    for i in range(data_save.shape[0]):  # per channel
        for j in range(DAS_reconstructions.shape[0] - 1):  # per sample
            data_save[i, j * timesamples: (j + 1) * timesamples] = DAS_reconstructions[j, i, :]
    z = int(data_save.shape[1] / timesamples)
    x = z * timesamples
    y = data_save.shape[1] - x
    data_save[:, x: data_save.shape[1]] = DAS_reconstructions[DAS_reconstructions.shape[0] - 1, :, timesamples - y: timesamples]

    return data_save

def plot_das_data(data):
    channels = data.shape[0]
    alpha=0.7
    i = 0
    plt.figure(figsize=(20, 12))
    for ch in range(channels):
        plt.plot(data[ch][:] + 12 * i, "-k", alpha=alpha)
        i += 1
    plt.show()

def deal_with_artifacts(data, filler = 0, Nt=1024):

    n_edges = int(data.shape[0] / Nt)

    for i in range(n_edges): # for every edge
        for j in range(data.shape[1]): # for every channel
            for n in range(5):
                data[Nt * i + n, j] = filler
                data[Nt * i - (n + 1), j] = filler

    return data


#model_names = os.listdir('experiments')
model_names = ["01_ablation_horizontal", "02_ablation_vertical", "03_accumulation_horizontal", "04_accumulation_vertical", "05_combined200", "06_combined800", "09_borehole_seismometer"]
data_type = "from_seis"

data_path = "data/synthetic_DAS/"+data_type+"/"
data_files = os.listdir(data_path)

for model_name in model_names:

    print("##############################################################")
    print("##############################################################")
    print("############### " + model_name + " ####################")
    print("##############################################################")
    print("##############################################################")

    # 1. Load model
    model = keras.models.load_model("experiments/" + model_name + "/" + model_name + ".h5")

    for data_file in data_files:

        # 2. Load data
        raw_data = np.load(data_path+data_file)

        # 3. Denoise data
        denoised_data = denoise_file(raw_data, model)

        #plot_das_data(raw_data)
        #plot_das_data(denoised_data)

        saving_path = "experiments/" + model_name + "/denoised_" + data_path[5:]
        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)

        np.save(saving_path + "/denoised_" + data_file.split("/")[-1], denoised_data)






