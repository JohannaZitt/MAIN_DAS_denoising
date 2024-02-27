import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from models import UNet, CallBacks, DataGenerator, DataGeneratorSeismometer
from datetime import date, timedelta
from models import seed
import random as python_random

"""

Main training on seismometer data or synthetic data


"""

# NumPy (random number generator used for sampling operations)
tf.random.set_seed(seed)
python_random.seed(seed)
rng = np.random.default_rng(seed)
tf.config.threading.set_inter_op_parallelism_threads(4)

""" Parameters """
N_sub = 8
batch_size = 32
Nt = 1024
N_epoch = 1500
batch_multiplier = 10 # set to 10 for 200 training samples (117*32 = 3744)
kernels = [(3,5), (5, 20), (8, 50)]

for kernel in kernels:

    model_params = {
        "use_bn": False, # batch normalization
        "use_dropout": False,
        "dropout_rate": 0.1,
        "N_blocks": 4,
        "f0": 4, # dimension of output space (i.e. the number of output filters in the convolution)
        "LR": 1e-4, # learning rate
        "data_shape": (N_sub, Nt, 1),
        "kernel": kernel,
        "AA": True # anti aliasing
    }

    print(model_params["kernel"])

    different_training_data = ["09_borehole_seismometer"]

    for training_data in different_training_data:

        """ Saving Paths """
        cwd = os.getcwd()
        savedir = os.path.join("experiments", training_data + "_" + str(kernel))
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        """Saving Path log"""
        logdir = os.path.join("experiments", training_data + "_" + str(kernel), "logs")
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        """ Callbacks """
        tensorboard_callback = CallBacks.tensorboard(logdir)
        checkpoint_callback = CallBacks.checkpoint(os.path.join(savedir, training_data + "_" + str(kernel) + ".h5"))

        """ Trainingdata """
        data_file = os.path.join(cwd, "data", "training_data", "preprocessed_seismometer", training_data + ".npy")

        # Load data
        data = np.load(data_file)
        N_ch, N_t = data.shape

        print(data.shape)

        #t_slice = slice(N_t//4, 3*N_t//4)
        #scaled_data = np.zeros_like(data)

        # normalize data
        #for i, wv in enumerate(data):
        #    scaled_data[i] = wv / wv[t_slice].std()

        # Split data 80-20 train-test
        split = int(0.8 * N_ch)
        train_data = data[:split]
        test_data = data[split:]

        print("Preparing masks")
        train_generator = DataGeneratorSeismometer(X=train_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
        test_generator = DataGeneratorSeismometer(X=test_data, Nt=Nt, N_sub=N_sub, batch_size=batch_size, batch_multiplier=batch_multiplier)
        print("Done")


        # visualize training data:
        """
        for i in range(10):
            for j in range(8):
                plt.plot(train_generator.samples[i][j] + 12*j, color="black", alpha=0.5)
            plt.show()
        """


        # Construct model
        net = UNet()
        net.set_params(model_params)
        model = net.construct()
        # model.summary()

        # Model training
        start = time.time()
        print("Start Model Training")
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
        x = dur_str.split(":")
        output_text = training_data + ": " + str(x[0]) + " Stunden, " + str(x[1]) + " Minuten und " + str(x[2]) + " Sekunden"

        with open("runtimes.txt", "a") as file:
            file.write("\n" + output_text)

        print(output_text)
