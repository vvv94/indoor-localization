from os import environ
from sys import path

# Global Settings
path.append("..")

from tools.utils import Utilities, NormY
from numpy import reshape, sqrt, mean, square, hstack, array, std
from random import seed
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MSE, MAE, MAPE, MSLE
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization, Activation
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

class Network():

    def __init__(self, fig_path, model_path, epochs=10, batch_size=64, loss='mse', metric='accuracy', dropout_rate=0.7, lr=0.001, wifi_b=2.71828, bt_b=2.71828, _seed_=666):

        # Preprocessing Parameters
        self.normY = NormY()
        self.rssi_min = 109
        self.wifi_b = wifi_b
        self.bt_b = bt_b

        # Model Required Directories
        self.fig_dir = fig_path
        self.model_path = model_path + 'model.h5'
        self.x_model_path = model_path + 'x_model.h5'
        self.y_model_path = model_path + 'y_model.h5'
        self.x_fig_dir = model_path + 'x_'
        self.y_fig_dir = model_path + 'y_'

        # Features
        self.wifi_features = 11
        self.bt_features = 46
        self.classes = 2

        # Train/Evaluate Parameters
        self.epochs=epochs
        self.batch_size = batch_size

        # Hyperparameters
        self.lr=lr
        self.dropout=dropout_rate
        self.loss = loss
        self.activation = 'elu'
        self.metric = metric

        # Initialize Keras
        clear_session()
        seed(_seed_)
        tf.random.set_seed(_seed_)

    def get_wifi_network(self):

        model = Sequential()
        model.add(Input(shape=(self.wifi_features,)))
        #model.add(Dense(units=64, activation=self.activation))
        model.add(Reshape((model.output_shape[1], 1)))
        model.add(Conv1D(filters=99, kernel_size=4, activation=self.activation, ))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=66, kernel_size=4, activation=self.activation, ))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=33, kernel_size=5, activation=self.activation, ))

        return model

    def get_bt_network(self):

        model = Sequential()
        model.add(Input(shape=(self.bt_features,)))
        #model.add(Dense(units=64, activation=self.activation))
        model.add(Reshape((model.output_shape[1], 1)))
        model.add(Conv1D(filters=99, kernel_size=16, activation=self.activation, ))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=66, kernel_size=16, activation=self.activation, ))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=33, kernel_size=16, activation=self.activation, ))

        return model

    def get_network(self, outputs=2, print_parameters=False):

        # Load separate models
        wifi_model = self.get_wifi_network()
        bt_model = self.get_bt_network()

        # Concatenate models
        mergedOut = Add()([wifi_model.output,bt_model.output])
        mergedOut = Flatten()(mergedOut)
        mergedOut = Dense(units=outputs, activation='elu')(mergedOut)
        model = Model([wifi_model.input,bt_model.input], mergedOut)
        model.compile(optimizer=Adam(lr=self.lr),loss=self.loss, metrics=[self.metric])

        if print_parameters:
            print(model.summary())

        return model

    def preprocess(self, train_set, validation_set, validate=False):

        # Split Input Data
        train_wifi_x, train_bt_x, train_labels = train_set
        val_wifi_x, val_bt_x, validation_labels = validation_set

        if validate:
            print(type(train_wifi_x))
            print(train_wifi_x.shape)
            print(type(train_bt_x))
            print(train_bt_x.shape)
            print(type(train_labels))
            print(train_labels.shape)
            print(type(val_wifi_x))
            print(val_wifi_x.shape)
            print(type(val_bt_x))
            print(val_bt_x.shape)
            print(type(validation_labels))
            print(validation_labels.shape)

        # Train Set
        self.normY.fit(train_labels[:, 0], train_labels[:, 1])
        train_wifi_x = Utilities.normalizeX(data=train_wifi_x, size=self.wifi_features ,exponent=self.wifi_b, limit=self.rssi_min)
        train_bt_x = Utilities.normalizeX(data=train_bt_x, size=self.bt_features, exponent=self.bt_b, limit=self.rssi_min, wifi=False)
        x_labels, y_labels = self.normY.normalizeY(long_data=train_labels[:, 0], lati_data=train_labels[:, 1])

        # Validation Set
        self.normY.fit(validation_labels[:, 0], validation_labels[:, 1]) if validate else self.normY.fit(train_labels[:, 0], train_labels[:, 1])
        val_wifi_x = Utilities.normalizeX(data=val_wifi_x, size=self.wifi_features, exponent=self.wifi_b, limit=self.rssi_min) if validate else []
        val_bt_x = Utilities.normalizeX(data=val_bt_x, size=self.bt_features, exponent=self.bt_b, limit=self.rssi_min, wifi=False) if validate else []
        val_x_labels, val_y_labels = self.normY.normalizeY(long_data=validation_labels[:, 0], lati_data=validation_labels[:, 1]) if validate else (array([]),array([]))

        # Combine into train/test sets
        train_set = (train_wifi_x, train_bt_x, hstack([x_labels, y_labels]))
        validation_set = (val_wifi_x, val_bt_x, hstack([val_x_labels, val_y_labels]))

        return  train_set, validation_set

    def fit(self, train_set, validation_set, validate=False, verbose=0, plot_training=False):

        # Data pre-processing
        train_set, validation_set = self.preprocess(train_set, validation_set, validate)

        # Load Model
        model = self.get_network(outputs=self.classes, print_parameters=False)

        # Train Model
        history = model.fit(x=[train_set[0],train_set[1]],
                            y=train_set[2],
                            batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        #Save Model
        model.save(self.model_path)

        if plot_training:
            self.plot(history, self.fig_dir)

    def predict(self, test_measurements):

        # Convert Input
        test_wifi_measurements = Utilities.normalizeX(data=test_measurements[0], size=self.wifi_features, exponent=self.wifi_b, limit=self.rssi_min)
        test_bt_measurements = Utilities.normalizeX(data=test_measurements[1], size=self.bt_features, exponent=self.bt_b, limit=self.rssi_min, wifi=False)

        # Load model
        model = load_model(self.model_path)

        # Calculate Prediction
        prediction = model.predict(x=[test_wifi_measurements,test_bt_measurements], batch_size=self.batch_size, verbose=0)

        # Normalize Output
        x_coords, y_coords = self.normY.reverse_normalizeY(prediction[:, 0],prediction[:, 1])

        return (x_coords.flatten(), y_coords.flatten())

    def error(self, test_measurements, test_labels, print_errors=True, separate=False):

        # Extract x,y points for test set
        real_x_pos, real_y_pos = test_labels[:, 0],test_labels[:, 1]

        # Get Predictions
        if separate:
            x_pos, y_pos = self.predict_separated(test_measurements=test_measurements)
        else:
            x_pos, y_pos = self.predict(test_measurements=test_measurements)

        # Print Output
        if print_errors:
            self.print_errors(x_pos,y_pos,real_x_pos,real_y_pos,0,limit=2.5)

        # Calculate X and Y Errors/ Mean Error/ STD of Errors
        x_error,        y_error,        total_error = sqrt(square(x_pos - real_x_pos)), sqrt(square(y_pos - real_y_pos)), sqrt(square(x_pos - real_x_pos) + square(y_pos - real_y_pos))
        m_x_error,      m_y_error,      m_total_error = mean(x_error), mean(y_error), mean(total_error)
        x_error_std,    y_error_std,    total_error_std = std(x_error), std(y_error), std(total_error)

        return  m_x_error, x_error_std, m_y_error, y_error_std, m_total_error, total_error_std

    def get_pseudolabels(self,test_measurements):

        # Convert Input
        test_wifi_measurements = Utilities.normalizeX(data=test_measurements[0], size=self.wifi_features, exponent=self.wifi_b, limit=self.rssi_min)
        test_bt_measurements = Utilities.normalizeX(data=test_measurements[1], size=self.bt_features, exponent=self.bt_b, limit=self.rssi_min, wifi=False)

        # Load model
        model = load_model(self.model_path)

        # Derive Prediction
        prediction = model.predict(x=test_measurements, batch_size=self.batch_size, verbose=0)

        # Normalize Output
        x_coords, y_coords = self.normY.reverse_normalizeY(prediction[:, 0],prediction[:, 1])

        # Zip to prediction and convert to proper format
        pseudo_labels = array(list(zip(x_coords, y_coords)))

        return reshape(pseudo_labels,(-1,2))

    def plot(self, history, fig_dir):

        # accuracy Plot
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(fname=fig_dir+'accuracy.png',dpi=1000)
        plt.close()

        # Loss Plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(fname=fig_dir+'loss.png',dpi=1000)
        plt.close()

    def print_errors(self,x_pred,y_pred,x_real,y_real,length, limit=2.0):

        large_x_error, large_y_error, outliers = 0,0,0

        for idx,(xp,yp,x,y) in enumerate(zip(x_pred,y_pred,x_real,y_real)):

            if idx < length :
                print('Predicted Point  : ' + str(xp)           +   ', \t' + str(yp)    + '\n'
                    + 'Actual Point     : ' + str(x)            +   ', \t' + str(y)     + '\n'
                    + 'Difference       : ' + str(round(abs(xp-x),4))    +   ', \t' + str(round(abs(yp-y),4))  + '\n')

            if round(abs(xp-x),4) > limit:
                large_x_error+=1
            if round(abs(yp-y),4) > limit:
                large_y_error+=1
            if round(abs(xp-x),4) > limit and round(abs(yp-y),4) > limit:
                outliers+=1

        print(large_x_error,large_y_error,outliers)

    def get_coord_network(self):

        # Wifi Model
        wifi_model = Sequential()
        wifi_model.add(Input(shape=(self.wifi_features,)))
        wifi_model.add(Reshape((wifi_model.output_shape[1], 1)))
        wifi_model.add(Conv1D(filters=99, kernel_size=4, activation=self.activation, ))
        wifi_model.add(Dropout(self.dropout))
        wifi_model.add(Conv1D(filters=66, kernel_size=4, activation=self.activation, ))
        wifi_model.add(Dropout(self.dropout))
        wifi_model.add(Conv1D(filters=33, kernel_size=5, activation=self.activation, ))

        # BT Model
        bt_model = Sequential()
        bt_model.add(Input(shape=(self.bt_features,)))
        bt_model.add(Reshape((bt_model.output_shape[1], 1)))
        bt_model.add(Conv1D(filters=99, kernel_size=16, activation=self.activation, ))
        bt_model.add(Dropout(self.dropout))
        bt_model.add(Conv1D(filters=66, kernel_size=16, activation=self.activation, ))
        bt_model.add(Dropout(self.dropout))
        bt_model.add(Conv1D(filters=33, kernel_size=16, activation=self.activation, ))

        # Concatenated Output
        output = Add()([wifi_model.output,bt_model.output])
        output = Flatten()(output)
        output = Dense(units=1, activation='elu')(output)

        # Create Final Model
        model = Model([wifi_model.input,bt_model.input], output)
        model.compile(optimizer=Adam(lr=self.lr),loss=self.loss, metrics=[self.metric])

        return model

    def fit_separated(self, train_set, validation_set, validate=False, verbose=0, plot_training=False):

        # Data pre-processing
        train_set, validation_set = self.preprocess(train_set, validation_set, validate)

        # Split labels in X-Y dimensions
        x_labels, y_labels = train_set[2][:,0], train_set[2][:,1]
        """
        print('|------------------------------------|')
        print(train_set[0][:,0])
        print('|------------------------------------|')
        print(train_set[0][:,1])
        print('|------------------------------------|')
        print(train_set[2][:,0])
        print('|------------------------------------|')
        print(train_set[2][:,1])
        print('|------------------------------------|')
        """
        # Load Model
        x_model = self.get_network(outputs=1, print_parameters=False)
        y_model = self.get_network(outputs=1, print_parameters=False)

        # Train Model
        x_history = x_model.fit(x=[train_set[0],train_set[1]], y=x_labels, batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)
        y_history = y_model.fit(x=[train_set[0],train_set[1]], y=y_labels, batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        #Save Model
        x_model.save(self.x_model_path)
        y_model.save(self.y_model_path)

        if plot_training:
            self.plot(history=x_history, fig_dir=self.x_fig_dir)
            self.plot(history=y_history, fig_dir=self.y_fig_dir)

    def predict_separated(self, test_measurements):

        # Convert Input
        test_wifi_measurements = Utilities.normalizeX(data=test_measurements[0], size=self.wifi_features, exponent=self.wifi_b, limit=self.rssi_min)
        test_bt_measurements = Utilities.normalizeX(data=test_measurements[1], size=self.bt_features, exponent=self.bt_b, limit=self.rssi_min, wifi=False)

        # Load model
        x_model = load_model(self.x_model_path)
        y_model = load_model(self.y_model_path)

        # Calculate Prediction
        x_prediction = x_model.predict(x=[test_wifi_measurements,test_bt_measurements], batch_size=self.batch_size, verbose=0)
        y_prediction = y_model.predict(x=[test_wifi_measurements,test_bt_measurements], batch_size=self.batch_size, verbose=0)

        # Normalize Output
        x_coords, y_coords = self.normY.reverse_normalizeY(x_prediction, y_prediction)

        return (x_coords.flatten(), y_coords.flatten())
