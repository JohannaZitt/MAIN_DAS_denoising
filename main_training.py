import os
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

""" Parameters """
N_sub = 11
batch_size = 32
Nt = 1024
N_epoch = 2000
batch_multiplier = 5 # set to 15 for 120 training samples (105*32 = 3360), set to 5 for 480 training samples (32*140 = 4480 samples per epoch)
model_params = {
    'use_bn': False, # batch normalization
    'use_dropout': False,
    'dropout_rate': 0.1,
    'N_blocks': 4,
    'f0': 4, # dimension of output space (i.e. the number of output filters in the convolution)
    'LR': 1e-4, # learning rate
    'data_shape': (N_sub, Nt, 1),
    'kernel': (3, 5),
    'AA': True # anti aliasing
}

different_training_data = ['08_combined480', '09_random480']

for training_data in different_training_data:

    """ Saving Paths """
    cwd = os.getcwd()
    savedir = os.path.join('experiments', training_data)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    """Saving Path log"""
    logdir = os.path.join('experiments', training_data, 'logs')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    """ Callbacks """
    tensorboard_callback = CallBacks.tensorboard(logdir)
    checkpoint_callback = CallBacks.checkpoint(os.path.join(savedir, training_data + '.h5'))

    """ Trainingdata """
    data_file = os.path.join(cwd, 'data', 'training_data', 'preprocessed_seismometer', training_data + '.npy')

    # Load data
    data = np.load(data_file)
    N_ch, N_t = data.shape
    print(data.shape)

    t_slice = slice(N_t//4, 3*N_t//4)
    scaled_data = np.zeros_like(data)

    # normalize data
    for i, wv in enumerate(data):
        scaled_data[i] = wv / wv[t_slice].std()

    # Split data 80-20 train-test
    split = int(0.8 * N_ch)
    train_data = scaled_data[:split]
    test_data = scaled_data[split:]

    print("Preparing masks")
    train_generator = DataGenerator(X=train_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
    test_generator = DataGenerator(X=test_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
    print("Done")

    print('Input: Data Shape: ', data.shape)
    print('Input: Train Data Shape: ', train_data.shape)
    print('Input: Test Data Shape: ', test_data.shape)

    print('Output: Train Samples Shape: ', train_generator.samples.shape)
    print('Output: Train Masks Shape: ', train_generator.masks.shape)
    print('Output: Train Masked_Samples Shape: ', train_generator.masked_samples.shape)

    print('Output: Test Samples Shape: ', test_generator.samples.shape)
    print('Output: Test Masks Shape: ', test_generator.masks.shape)
    print('Output: Test Masked_Samples Shape: ', test_generator.masked_samples.shape)

    # Construct model
    net = UNet()
    net.set_params(model_params)
    model = net.construct()
    # model.summary()

    # Model training
    start = time.time()
    print('Start Model Training')
    model.fit(
        x=train_generator,
        validation_data=test_generator,
        callbacks=[tensorboard_callback, checkpoint_callback],
        verbose=1, epochs=N_epoch,
    )

    # Generate output and measure runtime
    end = time.time()
    dur = end-start
    dur_str = str(timedelta(seconds=dur))
    x = dur_str.split(':')
    output_text = training_data + ': ' + str(x[0]) + ' Stunden, ' + str(x[1]) + ' Minuten und ' + str(x[2]) + ' Sekunden'

    with open('experiments/runtimes.txt', 'a') as file:
        file.write('\n' + output_text)

    print(output_text)