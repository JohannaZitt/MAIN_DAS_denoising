
import os

import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
from models import CallBacks, DataGeneratorDAS
import tensorflow as tf
from tensorflow import keras
from models import seed
tf.random.set_seed(seed)
import random as python_random
python_random.seed(seed)
""" Setting random seeds """
rng = np.random.default_rng(seed)

"""
Retraining on DAS data of the trained in main_training.py models
test
"""

""" Parameters """
N_sub = 11
batch_size = 32
batch_multiplier = 10
N_epoch = 200
pretrained_model_name = "05_combined200"
model_name = "07_retrained_combined200_2"
trainingdata = "data/training_data/preprocessed_DAS/retraining_data.npy"

# Load pretrained Model:
model_file = os.path.join("experiments", pretrained_model_name, pretrained_model_name + ".h5")
model = keras.models.load_model(model_file)

""" Callbacks """
logdir = os.path.join("experiments", model_name, "logs")
savefile = model_name + ".h5"
savedir = os.path.join("experiments", model_name)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

tensorboard_callback = CallBacks.tensorboard(logdir)
checkpoint_callback = CallBacks.checkpoint(os.path.join(savedir, savefile))


""" Load Data and Preprocess Data """
data = np.load(trainingdata)
N_ch = data.shape[1]

# train and test set:
test_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
train_indices = np.delete(np.arange(data.shape[0]), test_indices)
test_data = data[test_indices,:]
train_data = data[train_indices,:]

print("Preparing masks")
# Prepare data generator for the train and test set
train_generator = DataGeneratorDAS(X=train_data, N_sub=N_sub, batch_multiplier=batch_multiplier, batch_size=batch_size)
test_generator = DataGeneratorDAS(X=test_data, N_sub=N_sub, batch_multiplier=batch_multiplier, batch_size=batch_size)
print("Done")

#for i in range(630, 640):
#    for j in range(11):
#        plt.plot(train_generator.samples[i][j] + 8 * j, color="black", alpha=0.5)
#    plt.show()


start = time.time()

# Train the model:
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
x = dur_str.split(":")
print("Programm endete um: " + time.strftime("%H:%M:%S") + "; Laufzeit: " + str(x[0]) + " Stunden, " + str(x[1]) + " Minuten und " + str(x[2]) + " Sekunden")

output_text = "Retraining " + model_name + " took " + str(x[0]) + " hours, " + str(x[1]) + " minutes und " + str(x[2]) + " seconds"
with open("runtimes.txt", "a") as file:
    file.write('\n' + output_text)