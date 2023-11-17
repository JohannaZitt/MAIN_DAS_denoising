import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.optimizers import Adam
from scipy.signal import butter, filtfilt, tukey


'''

Code modified after van den Ende et al. (2021) "Self-Supervised Deep Learning Approach for Blind Denoising and 
Waveform Coherence Enhancement in Distributed Acoustic Sensing Data"

'''



""" Setting random seeds """
seed = 42

# TensorFlow
tf.random.set_seed(seed)

# Python
import random as python_random
python_random.seed(seed)

# NumPy (random number generator used for sampling operations)
rng = np.random.default_rng(seed)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low < 0:
        Wn = high
        btype = "lowpass"
    elif high < 0:
        Wn = low
        btype = "highpass"
    else:
        Wn = [low, high]
        btype = "bandpass"

    b, a = butter(order, Wn, btype=btype)

    return b, a

def taper_filter(arr, fmin, fmax, samp_DAS):
    b_DAS, a_DAS = butter_bandpass(fmin, fmax, samp_DAS)
    window_time = tukey(arr.shape[1], 0.1)
    arr_wind = arr * window_time
    arr_wind_filt = filtfilt(b_DAS, a_DAS, arr_wind, axis=-1)
    return arr_wind_filt

class DataGenerator(keras.utils.Sequence):

    def __init__(self, X, Nt=2048, N_sub=10, batch_size=32, batch_multiplier=10):
        # Data matrix
        self.X = X
        # Number of Trainingsamples
        self.Nx = X.shape[0]
        # Number of time sampling points in data
        self.Nt_all = X.shape[1]
        # Number of time sampling points in a slice
        self.Nt = Nt
        # Number of stations per batch sample
        self.N_sub = N_sub
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier

        self.on_epoch_end()

    def __len__(self):
        """ Number of mini-batches per epoch """
        return int(self.batch_multiplier * self.Nx * self.Nt_all / float(self.batch_size * self.Nt))

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        pass

    def __getitem__(self, idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        selection = slice(idx * batch_size, (idx + 1) * batch_size)
        samples = np.expand_dims(self.samples[selection], -1)
        masked_samples = np.expand_dims(self.masked_samples[selection], -1)
        masks = np.expand_dims(self.masks[selection], -1)
        return (samples, masks), masked_samples

    def __data_generation(self):
        """ Generate a total batch """
        # Number of mini-batches
        N_batch = self.__len__()
        N_total = N_batch * self.batch_size
        Nt = self.Nt
        N_sub = self.N_sub
        Nt_all = self.Nt_all
        # Buffer for mini-batches
        samples = np.zeros((N_total, N_sub, Nt))
        # Buffer for masks
        masks = np.ones_like(samples)

        t_starts = rng.integers(low=1, high=Nt_all-Nt, size=N_total)
        X = self.X

        # we want to detect stick-slip events the most -> use velocities of p- and s- waves:
        # p-wave velocity: 3600-3900 m/s
        # s-wave velocity: 1700-1950 m/s
        # rayleigh wave velocity: 1650 - 1668 m/s
        # -> between 1650 - 3900
        s_min = 1 / 3900.
        s_max = 1 / 1650.

        gauge = 10.
        fs = 400.

        log_SNR_min = -2
        log_SNR_max = 4

        # Loop over samples
        for s, t_start in enumerate(t_starts):
            sample_ind = rng.integers(low=0, high=self.Nx)
            t_slice = slice(t_start, t_start + Nt)

            # Time reversal
            order = rng.integers(low=0, high=2) * 2 - 1
            # Polarity flip
            sign = rng.integers(low=0, high=2) * 2 - 1
            # Move-out direction
            direction = rng.integers(low=0, high=2) * 2 - 1

            slowness = rng.random() * (s_max - s_min) + s_min
            shift = direction * gauge * slowness * fs # shift is in the range of [-6, 6] when s_min = 1 / 3900 and s_max = 1 / 1650
            sample = sign * X[sample_ind, ::order] # without timereversals: sample = sign * X[sample_ind, :]
            # (die channel spacing length ist aufm Rhonegletscher 4m hier werden Entfernungen je nach Wellentyp zwischen 4m und 60m angenommen)


            SNR = rng.random() * (log_SNR_max - log_SNR_min) + log_SNR_min
            SNR = 10 ** (0.5 * SNR)
            amp = 2 * SNR / np.abs(sample).max()
            sample = sample * amp

            for i in range(N_sub):
                samples[s, i] = np.roll(sample, int(i * shift))[t_slice]

            # Select one waveform to blank
            blank_ind = rng.integers(low=0, high=self.N_sub)
            masks[s, blank_ind] = 0

        gutter = 100
        noise = rng.standard_normal((N_total * N_sub, Nt + 2 * gutter))
        noise = taper_filter(noise, fmin=1.0, fmax=120.0, samp_DAS=fs)[:, gutter:-gutter]
        noise = noise.reshape(*samples.shape)

        noisy_samples = samples + noise
        for s, sample in enumerate(noisy_samples):
            noisy_samples[s] = sample / sample.std()

        self.samples = noisy_samples
        self.masks = masks
        self.masked_samples = noisy_samples * (1 - masks)
        pass

class CallBacks:

    @staticmethod
    def tensorboard(logdir):
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir,
            write_graph=True,
            write_images=True,
            profile_batch=0,
            update_freq="epoch",
            histogram_freq=0,
        )
        return tensorboard_callback

    @staticmethod
    def checkpoint(savefile):
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            savefile,
            verbose=0,
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            mode="auto",
            update_freq="epoch",
        )
        return checkpoint_callback


class UNet:

    def __init__(self):
        self.kernel = (3, 5)
        self.f0 = 2
        self.N_blocks = 4
        self.use_bn = True
        self.use_dropout = True
        self.dropout_rate = 0.1
        self.AA = False
        self.LR = 5e-4
        self.initializer = keras.initializers.Orthogonal(seed=seed)
        self.activation = tf.keras.activations.swish
        self.data_shape = (10, 2048, 1)
        pass

    def set_params(self, params):
        """
        Update model parameters
        """
        self.__dict__.update(params)
        pass

    def conv_layer(self, x, filters, kernel_size,
                   use_bn=False, use_dropout=False, activ=None):
        """
        Convolution layer > batch normalisation > activation > dropout
        """
        use_bias = True
        if use_bn:
            use_bias = False

        x = Conv2D(
            filters=filters, kernel_size=kernel_size, padding="same",
            activation=None, kernel_initializer=self.initializer,
            use_bias=use_bias
        )(x)

        if use_bn:
            x = BatchNormalization()(x)

        if activ is not None:
            x = Activation(activ)(x)

        if use_dropout:
            x = GaussianDropout(self.dropout_rate)(x)

        return x
    
    def MaxBlurPool(self, x, kernel_size=(1, 4)):
        
        if kernel_size[1] == 1:
            a = np.array([1.,])
        elif kernel_size[1] == 2:
            a = np.array([1., 1.])
        elif kernel_size[1] == 3:
            a = np.array([1., 2., 1.])
        elif kernel_size[1] == 4:    
            a = np.array([1., 3., 3., 1.])
        elif kernel_size[1] == 5:    
            a = np.array([1., 4., 6., 4., 1.])
        elif kernel_size[1] == 6:    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif kernel_size[1] == 7:    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        
        a = a / a.sum()
        a = np.repeat(a, x.shape[-1]*x.shape[-1])
        a = a.reshape((kernel_size[0], kernel_size[1], x.shape[-1], x.shape[-1]))
        
        x = MaxPool2D(pool_size=kernel_size, strides=(1, 1))(x)
        x = tf.nn.conv2d(input=x, filters=a, strides=kernel_size, padding="SAME")
        
        return x

    def construct(self):
        """
        Construct UNet model
        """

        f = self.f0
        kernel = self.kernel
        use_bn = self.use_bn
        use_dropout = self.use_dropout
        AA = self.AA
        activation = self.activation
        data_shape = self.data_shape

        input = Input(data_shape)
        mask_input = Input(data_shape)
        c_mask_input = 1 - mask_input
        x = mask_input * input

        """ Encoder """
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)

        x_prev = [x]

        for i in range(self.N_blocks):
            
            if AA:
                x = self.MaxBlurPool(x, kernel_size=(1, 4))
            else:
                x = MaxPool2D(pool_size=(1, 4))(x)
            
            f = f * 2
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)
            x_prev.append(x)

        """ Decoder """
        for i in range(self.N_blocks-1):
            x = UpSampling2D(size=(1, 4), interpolation="bilinear")(x)
            f = f // 2
            x = concatenate([x, x_prev[-(i+2)]])
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)
            x = self.conv_layer(x, filters=f, kernel_size=kernel,
                                use_bn=use_bn, use_dropout=use_dropout,
                                activ=activation)

        x = UpSampling2D(size=(1, 4), interpolation="bilinear")(x)  # 128
        f = f // 2
        x = concatenate([x, x_prev[0]])
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)
        x = self.conv_layer(x, filters=f, kernel_size=kernel,
                            use_bn=use_bn, use_dropout=use_dropout,
                            activ=activation)
        x = self.conv_layer(x, filters=1, kernel_size=kernel,
                            use_bn=False, use_dropout=False, activ=None)

        x = c_mask_input * x
        model = Model([input, mask_input], x)
        model.build(input_shape=[data_shape, data_shape])

        # Build and generate a summary
        # model.summary()

        # Train auto-encoder
        model.compile(
            optimizer=Adam(learning_rate=self.LR),
            loss="mean_squared_error",
        )

        return model
