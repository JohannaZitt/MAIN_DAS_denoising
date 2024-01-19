import numpy as np
import h5py
import keras
import gc
import os
from pydas_readers.readers import load_das_h5_CLASSIC as load_das_h5, write_das_h5
from scipy.signal import butter, lfilter
import time
from datetime import timedelta


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def resample(data, ratio):
    # spatial axis:
    n_ch = data.shape[0]
    if n_ch == 4800 or n_ch == 4864:
        data = data[::4, :]
    else:
        data = data[::2, :]

    # time axis
    res = np.zeros((data.shape[0], int(data.shape[1] / ratio)))
    for i in range(data.shape[0]):
        res[i] = np.interp(np.arange(0, len(data[0]), ratio), np.arange(0, len(data[0])), data[i])

    return res

def denoise_file (file, timesamples, model, N_sub, fs_trainingdata):

    # load data
    with h5py.File(file, "r") as f:
        DAS_data = f["Acoustic"][:]

    # load headers
    headers = load_das_h5.load_headers_only(file)

    # reshape data
    DAS_data = DAS_data.T
    DAS_data = DAS_data.astype('f')
    # resample data
    print(DAS_data.shape)
    DAS_data = resample(DAS_data, headers['fs'] / 400)
    print(DAS_data.shape)
    # preprocessing
    lowcut: int = 1
    highcut = 120
    for i in range(DAS_data.shape[0]):
        # filter data
        DAS_data[i] = butter_bandpass_filter(DAS_data[i], lowcut, highcut, fs=400, order=4)
        # normalize data
        DAS_data[i] = DAS_data[i] / np.abs(DAS_data[i]).max()

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
    data_save = data_save.T

    # Update Headers:
    headers['npts'] = data_save.shape[0]
    headers['nchan'] = data_save.shape[1]
    headers['fs'] = fs_trainingdata
    headers['dx'] = 8

    return data_save, headers

def deal_with_artifacts(data, filler = 0, Nt=1024):

    n_edges = int(data.shape[0] / Nt)

    for i in range(n_edges): # for every edge
        for j in range(data.shape[1]): # for every channel
            for n in range(5):
                data[Nt * i + n, j] = filler
                data[Nt * i - (n + 1), j] = filler

    return data


models_path = 'experiments'
model_names = os.listdir(models_path)
model_names = ['01_ablation_horizontal']

raw_DAS_path = 'data/raw_DAS'
data_types = ['test_ablation_RA87']

n_sub = 11
timesamples = 1024
fs_trainingdata = 400
DEAL_WITH_ARTIFACTS = True

# every model:
for model_name in model_names:

    # for every raw data folder
    for data_type in data_types:

        raw_das_folder_path = os.path.join(raw_DAS_path, data_type)
        saving_path = os.path.join('experiments', model_name, 'denoisedDAS', data_type)
        model_file = os.path.join('experiments', model_name, model_name + '.h5')

        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)

        files = os.listdir(raw_das_folder_path)

        for file in files:
            start = time.time()
            print('------------ File: ', file, ' is denoising --------------------')

            # Path to raw DAS data
            raw_das_file_path = os.path.join(raw_das_folder_path, file)
            saving_filename = '/denoised_' + file

            # load model
            model = keras.models.load_model(model_file)
            # denoise data:
            if not os.path.exists(saving_path+saving_filename):
                data, headers = denoise_file(file=raw_das_file_path, timesamples=timesamples, model=model, N_sub=n_sub, fs_trainingdata=fs_trainingdata)

                # deal with artifacts:
                if DEAL_WITH_ARTIFACTS:
                    print('deals with artifacts')
                    data = deal_with_artifacts(data)
                write_das_h5.write_block(data, headers, saving_path + saving_filename)
                print('Saved Data with Shape: ', data.shape)

                # Measuring time for one file
                end = time.time()
                dur = end - start
                dur_str = str(timedelta(seconds=dur))
                x = dur_str.split(':')
                print('Laufzeit für file ' + str(file) + ': ' + str(x[1]) + ' Minuten und ' + str(x[2]) + ' Sekunden.')

            else:
                print(saving_filename, ' does exist already. No Denoising is performed. ')

            gc.collect()
