import numpy as np
import h5py
import keras
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
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
    res = np.zeros((data.shape[0], int(data.shape[1]/ 2.5)))
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

    # preprocessing
    lowcut: int = 1
    highcut = 120
    for i in range(DAS_data.shape[0]):
        # filter data
        butter_bandpass_filter(DAS_data[i], lowcut, highcut, fs=1000, order=4)
        # normalize data
        DAS_data[i] = DAS_data[i] / DAS_data[i].std()

    # resampling auf 400Hz bringen. Hier gibt es teilweise auch andere sample frequencies!!!
    n_ch, n_t = DAS_data.shape
    print('DAS_data.shape vor resampels: ', DAS_data.shape)
    DAS_data = resample(DAS_data, headers['fs']/400)
    print('DAS_data.shape nach resampels: ', DAS_data.shape)

    # Lieber immer alles denoisen!! -> auch besser für plotten
    #if n_ch == 4800 or n_ch == 4864:
    #    DAS_data=DAS_data[::2,:]

    n_ch, n_t = DAS_data.shape
    n_samples = int(n_t / timesamples) + 1
    data = np.zeros((n_samples, n_ch, timesamples))
    for i in range(n_samples - 1):
        data[i] = DAS_data[:, i * timesamples: (i + 1) * timesamples]
    data[n_samples - 1] = DAS_data[:, n_t - timesamples: n_t]

    DAS_reconstructions = np.zeros_like(data)
    N_samples, N_ch, N_t = data.shape

    # loop over every batche of 2048 sampling points
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

        # Create J-invariant reconstructions
        results = model.predict((eval_samples, masks))
        DAS_reconstructions[n] = np.sum(results, axis=1)[:, :, 0]

    # reshape data into original shape
    data_save = np.zeros_like(DAS_data)
    for i in range(data_save.shape[0]):  # per channel
        for j in range(DAS_reconstructions.shape[0] - 1):  # per sample
            data_save[i, j * timesamples: (j + 1) * timesamples] = DAS_reconstructions[j, i, :]

    # Der DAS Datensatz der denoised werden soll, hat die Größ von [2560, 12.000]
    # 12000 modulo 2048 =  1760
    # 12000 div 2048 = 5 Rest 1760

    z = int(data_save.shape[1] / timesamples)

    x = z * timesamples
    y = data_save.shape[1] - x
    data_save[:, x: data_save.shape[1]] = DAS_reconstructions[DAS_reconstructions.shape[0] - 1, :, timesamples - y: timesamples]
    data_save = data_save.T

    # Update Headers:
    headers['npts'] = data_save.shape[0]
    headers['nchan'] = data_save.shape[1]
    headers['fs'] = fs_trainingdata
    headers['dx'] = 4

    #print('Data_Shape: ' + str(data_save.shape))

    return data_save, headers

def deal_with_artifacts(data, filler = 0, Nt=1024):

    n_edges = int(data.shape[0] / Nt)

    for i in range(n_edges): # for every edge
        for j in range(data.shape[1]): # for every channel
            for n in range(7):
                data[Nt * i + n, j] = filler
                data[Nt * i - (n + 1), j] = filler

    return data



# TODO: GPU speed up wäre hier auch noch richtig nice! -> vorallem, wenn ganzer Datensatz denoist werden sollte.
# TODO: Denoisen für nur eine gewisse channel Anzahl

models=['01_ablation_horizontal']
n_sub = 11
timesamples = 1024
fs_trainingdata = 400
DEAL_WITH_ARTIFACTS = True

for model in models:

    raw_das_folder_path = "data/raw_DAS"
    model_name = model
    saving_path = os.path.join('experiments', model_name, 'denoisedDAS/')
    model_file = os.path.join('experiments', model_name, model_name + '.h5')

    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    files = os.listdir(raw_das_folder_path)



    for file in files:
        start = time.time()
        print('------------ File: ', file, ' is denoising --------------------')

        # Path to raw DAS data
        raw_das_file_path = os.path.join(raw_das_folder_path, file)
        saving_filename = 'denoised_' + file

        # config ansprechen:
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)
        #set_session(sess)

        # load model
        model = keras.models.load_model(model_file)
        data, headers = denoise_file(file=raw_das_file_path, timesamples=timesamples, model=model, N_sub=n_sub, fs_trainingdata=fs_trainingdata)
        if DEAL_WITH_ARTIFACTS:
            print('deals with artifacts')
            data = deal_with_artifacts(data)
        write_das_h5.write_block(data, headers, saving_path + saving_filename)
        print('Saved Data with Shape: ', data.shape)

        # reset GPU memory
        #sess.close()
        #tf.reset_default_graph()

        # Measuring time for one file
        end = time.time()
        dur = end - start
        dur_str = str(timedelta(seconds=dur))
        x = dur_str.split(':')
        print('Laufzeit für file ' + str(file) + ': ' + str(x[1]) + ' Minuten und ' + str(x[2]) + ' Sekunden.')
