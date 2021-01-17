from os import environ
from numpy import hstack, zeros, ones, asarray, hsplit, amin, amax, squeeze,arange, mean, transpose
from numpy.random import rand, randn, choice
from random import Random
from matplotlib import pyplot
from matplotlib.pyplot import scatter
from pandas import read_csv
# Global Settings
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, Huber, MSE, MAE
from tensorflow.keras.activations import sigmoid, softmax, elu, relu, selu
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_uniform
from tensorflow.config.experimental import (VirtualDeviceConfiguration, list_physical_devices, set_virtual_device_configuration)



class RSSI_GAN(object):

    def __init__(self, latents=5, inputs=57, outputs=2):

        self.latents = latents
        self.generator_input = latents
        self.generator_output = inputs + outputs
        self.discriminator_input = inputs + outputs
        self.discriminator_output = 1

        self.generator = self.generator(inputs=self.generator_input, outputs=self.generator_output)
        self.discriminator = self.discriminator(inputs=self.discriminator_input, outputs=self.discriminator_output)

        self.loc = squeeze(hsplit(asarray(read_csv("./data/Train.csv",sep=';',header = 0).iloc[:,-2:]),2))
        self.rssi = asarray(read_csv("./data/Train.csv",sep=';',header = 0).iloc[:,:-2])
        self.data = hstack((self.rssi,transpose(self.loc)))
        self.min = [amin(self.loc[0],axis=0), amin(self.loc[1],axis=0)]
        self.mac = [amax(self.loc[0],axis=0), amax(self.loc[1],axis=0)]

    def generator(self, inputs, outputs, loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'] ,activation='relu', fc_activation='linear', kernel_initializer='he_uniform'):

        generator = Sequential()
        generator.add(Dense(units=15, activation=activation, kernel_initializer=kernel_initializer, input_dim=inputs))
        generator.add(Dense(units=outputs, activation=fc_activation))
        return generator

    def discriminator(self, inputs, outputs, loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'] ,activation='relu', fc_activation='sigmoid', kernel_initializer='he_uniform'):

        discriminator = Sequential()
        discriminator.add(Dense(units=25, activation=activation, kernel_initializer=kernel_initializer, input_dim=inputs))
        discriminator.add(Dense(units=outputs, activation=fc_activation))
        discriminator.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return discriminator

    def get_model(self, loss='binary_crossentropy', optimizer=Adam(lr=0.001)):

        # Make weights in the discriminator not trainable
        self.discriminator.trainable = False

        # Create GAN model
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(loss= loss, optimizer=optimizer)

        return model

    def real_samples(self, n_samples=5):

        # Generate Data
        data  = self.data[choice(self.data.shape[0], size=n_samples, replace=False), :]
        input, output = data[:,0:-2], data[:,-2:]
        # Reshape
        input, output = input.reshape(n_samples, input.shape[1]) ,output.reshape(n_samples, output.shape[1])
        # Convert to train set/labels
        train_set, train_labels = hstack((input, output)), ones((n_samples, 1))

        return train_set, train_labels

    def fake_samples(self, model, n_samples=5):

        # Generate inputs
        input = randn(self.latents * n_samples)
        # Reshape inputs
        input = input.reshape(n_samples, self.latents)
        # Predict Outputs and form it train_set
        train_set, train_labels = model.predict(input), zeros((n_samples, 1))

        return train_set, train_labels

    def latents_inputs(self, n_samples):

        return randn(self.latents * n_samples).reshape(n_samples, self.latents)

    def train_eval(self, model, epochs=100, batch_size=114, eval_rate=10):

        # Load models
        h_batch = int(batch_size / 2)

        for i in range(epochs):
            ##########################################################################
            # Get Real/Fake Samples for Discriminator
            x_real, y_real = self.real_samples(n_samples=h_batch)
            x_fake, y_fake = self.fake_samples(model=self.generator, n_samples=h_batch)
            # Update Discriminator
            self.discriminator.train_on_batch(x_real, y_real)
            self.discriminator.train_on_batch(x_fake, y_fake)
            ##########################################################################
            # Get Samples for GAN
            samples, labels = self.latents_inputs(n_samples=batch_size), ones((batch_size, 1))
            # Update GAN from Discriminator's Error
            model.train_on_batch(samples, labels)
            ##########################################################################
            # Evaluate GAN
            if i % eval_rate == 0 and i!=0:
                self.evaluate(epoch=i, n_samples=30, last=False if i+1!=epochs else True)
            ##########################################################################

    def evaluate(self, epoch, n_samples, last=False):

        # Get Real/Fake Samples
        x_real, y_real = self.real_samples(n_samples=n_samples)
        x_fake, y_fake = self.fake_samples(model=self.generator, n_samples=n_samples)
        # Evaluate
        _, acc_real = self.discriminator.evaluate(x_real, y_real, verbose=0)
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # Plot Results
        print(epoch, acc_real, acc_fake)
        # Scatter Plot
        fig, ax = pyplot.subplots(2, figsize=(10, 6))
        #ax[0].scatter(x_real[:,-2], x_real[:,-1], color='red')
        ax[0].scatter(x_fake[:,-2], x_fake[:,-1], color='blue')
        ax[0].scatter(self.loc[0], self.loc[1], color='red')
        ax[0].set_xlabel("Pos X")
        ax[0].set_ylabel("Pos Y")
        ax[1].plot(mean(x_real[:,:-2],axis=0), color='red')
        ax[1].plot(mean(x_fake[:,:-2],axis=0), color='blue')
        ax[1].plot(mean(self.rssi, axis=0), color='green')
        ax[1].set_ylabel("RSSI Measurement")
        ax[1].set_xlabel("AP (1-11) AND BT (12-57)")
        pyplot.savefig('plot',dpi=1000)


if __name__ == "__main__":

    gpus = list_physical_devices("GPU")
    if not gpus:
        print("No GPU's available. Will run on CPU.")
    else:
        set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=10000)])
    
    # Create GAN
    GAN = RSSI_GAN(latents=120, inputs=57, outputs=2)
    # Load Model
    model = GAN.get_model()
    # Train/Evaluate Model
    GAN.train_eval(model,epochs=1000, batch_size=114, eval_rate=100)