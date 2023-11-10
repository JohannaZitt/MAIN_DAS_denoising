import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Gets all the warnings rid of in the terminal
import time
import tensorflow as tf

import shutil
import numpy as np
from models import UNet, CallBacks, DataGenerator
from datetime import date, timedelta
from models import seed
import random as python_random

'''

Main training of the network on seismometer data or synthetic data


'''

# NumPy (random number generator used for sampling operations)
tf.random.set_seed(seed)
python_random.seed(seed)
rng = np.random.default_rng(seed)
tf.config.threading.set_inter_op_parallelism_threads(4)

N_subs = [11]

for n in N_subs:
    print('Starting Training with n_sub = ' + str(n))

    """ Parameters """
    N_sub = n
    batch_size = 16 # mini-batch-size
    #model_name = day + "__" + folder + "__" + dataset[:-4]
    Nt = 1024
    N_epoch = 2000
    batch_multiplier = 15
    model_params = {
        "use_bn": False, #Use Batch Normalization
        "use_dropout": False, #Use Dropout
        "dropout_rate": 0.1, #Dropout Rate
        "N_blocks": 4, #Anzahl, woe viele Encoder und wie viele Decoder Blöcke es gibt
        "f0": 4, #Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
        "LR": 1e-4, #1e-4, #learning rate
        "data_shape": (N_sub, Nt, 1),
        "kernel": (3, 5),
        "AA": True # Anti Aliasing
    }


    """ Saving Paths """
    day = str(date.today())
    cwd = os.getcwd()
    model_name = day + '_Test16_' + str(N_sub) + '_' + str(Nt)

    savedir = os.path.join("experiments", model_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    logdir = os.path.join("experiments", model_name, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    """ Callbacks """
    tensorboard_callback = CallBacks.tensorboard(logdir)
    checkpoint_callback = CallBacks.checkpoint(os.path.join(savedir, model_name + ".h5"))

    """ Trainingdata """
    folder = "seismometer"
    dataset = "MAIN_trainingdata.npy"
    data_file = os.path.join(cwd, "data", "preprocessed_training_data", folder, dataset)

    # Load data
    data = np.load(data_file)
    N_ch, N_t = data.shape
    print(data.shape)

    t_slice = slice(N_t//4, 3*N_t//4)
    scaled_data = np.zeros_like(data)

    # Loop over data and normalise
    for i, wv in enumerate(data):
        scaled_data[i] = wv / wv[t_slice].std()

    #print('data_type: ', type(scaled_data[3]))
    #print('data_point: ', scaled_data[5:10])

    '''
    # View Training data
    
    fig, axes = plt.subplots(nrows=8, figsize=(9, 4), constrained_layout=True)
    
    # Loop over 5 waveforms
    for i in range(0, 7): #len(axes)
        axes[i].plot(scaled_data[i])
        axes[i].axis("off")
        print(max(data[i, :]))
    plt.show()
    '''


    # Split data 80-20 train-test
    split = int(0.8 * N_ch) # also bei input Datensatz größe = 21 -> 0.8 * N_ch = 16.8 = 16
    train_data = scaled_data[:split]
    test_data = scaled_data[split:]

    print("Preparing masks")
    # Prepare data generator for the train and test set
    train_generator = DataGenerator(X=train_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
    test_generator = DataGenerator(X=test_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
    print("Done")

    # Construct model
    net = UNet()
    net.set_params(model_params)
    model = net.construct()
    # model.summary()



    # View Trainingdata + artificial noise
    '''
    for i in range(0, 30):
        plt.plot(train_generator.samples[i][0])
        plt.show()
    '''


    # Model training
    start = time.time()
    print('Start Model Training')
    model.fit(
        x=train_generator,
        validation_data=test_generator,
        callbacks=[tensorboard_callback, checkpoint_callback],
        verbose=1, epochs=N_epoch,
    )

    # Generate Output and measure runtime
    end = time.time()
    dur = end-start
    dur_str = str(timedelta(seconds=dur))
    x = dur_str.split(':')
    print('Programm endete um: ' + time.strftime("%H:%M:%S") + '; Laufzeit: ' + str(x[0]) + ' Stunden, ' + str(x[1]) + ' Minuten und ' + str(x[2]) + ' Sekunden')