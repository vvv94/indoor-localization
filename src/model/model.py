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

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import clear_session


class EncoderDNN(object):

    def __init__(self, input, model_path, epochs=10, batch_size=64,loss='mse', metric='accuracy', optimizer='Adam', dropout_rate=0.7, lr=0.001, b=2.71828, _seed_=666):

        clear_session()
        seed(_seed_)
        tf.random.set_seed(_seed_)

        self.epochs=epochs
        self.batch_size = batch_size
        self.b=b

        self.features = input
        self.prediction = 2
        self.input = Input((input,))
        self.model_path = model_path + 'model.h5'
        self.normY = NormY()

        #Hyperparameters
        self.lr=lr
        self.dropout=dropout_rate
        self.loss = loss
        self.activation = 'elu'
        self.metric = metric

    def get_network(self):
        
        model = Sequential()
        model.add(Input(shape=(self.features,)))
        #model.add(Dense(units=64, activation=self.activation))
        model.add(Reshape((model.output_shape[1], 1)))
        #model.add(Conv1D(filters=128, kernel_size=15, activation=self.activation, ))
        #model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=64, kernel_size=13, activation=self.activation, ))        
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=32, kernel_size=11, activation=self.activation, ))        
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=16, kernel_size=11, activation=self.activation, ))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=8, kernel_size=12, activation=self.activation, ))
        model.add(Dropout(self.dropout))
        model.add(Conv1D(filters=4, kernel_size=13, activation=self.activation, ))
        model.add(Flatten())
        model.add(Dense(units=self.prediction, activation='elu'))
        model.compile(optimizer=Adam(lr=1e-3),loss=self.loss, metrics=[self.metric])
        #print(model.summary())
        print('Hello')
        return model 

    def preprocess(self, x, y, val_x, val_y, validate=False):

        norm_val_X, norm_val_long, norm_val_lati = [],[],[]
        norm_X = Utilities.normalizeX(data=x, size=self.features ,b=self.b)
        if validate:
            norm_val_X = Utilities.normalizeX(data=val_x, size=self.features, b=self.b)

        self.normY.fit(y[:, 0], y[:, 1])
        norm_long, norm_lati = self.normY.normalizeY(long_data=y[:, 0], lati_data=y[:, 1])
        if validate:
            norm_val_long, norm_val_lati = self.normY.normalizeY(long_data=val_y[:, 0], lati_data=val_y[:, 1])

        return  norm_X, norm_val_X, norm_long, norm_lati, norm_val_long, norm_val_lati

    def fit(self, x, y, val_x, val_y, test_x,test_y, validate=False, verbose=0):

        # Data pre-processing
        norm_X, norm_val_X, norm_long, norm_lati, norm_val_long, norm_val_lati = self.preprocess(x, y, val_x, val_y,validate)

        # Load Model
        model = self.get_network()

        # Train Model
        if validate:
            history = model.fit(norm_X, hstack([norm_long, norm_lati]),validation_data=(norm_val_X, hstack([norm_val_long, norm_val_lati])), epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
        else:
            history = model.fit(norm_X, hstack([norm_long, norm_lati]), epochs=self.epochs, batch_size=66, verbose=verbose)


        #Save Model
        model.save(self.model_path)

        return history

    def predict(self, x, batch_size):

        # Convert Input
        x = Utilities.normalizeX(data=x, size=self.features, b=self.b)

        # Load model
        model = load_model(self.model_path)

        # Derive Prediction
        predict_pos = model.predict(x, batch_size=batch_size)

        # Normalize Output
        predict_X, predict_Y = self.normY.reverse_normalizeY(predict_pos[:, 0],predict_pos[:, 1])

        return predict_X,predict_Y

    def error(self, test_input, real_pos):

        position = self.predict(x=test_input, batch_size=self.batch_size)

        pred_X = reshape(position[0],(1,len(position[0])))
        pred_Y = reshape(position[1],(1,len(position[1])))

        X_error = mean(sqrt(square(pred_X - real_pos[:, 0])))
        Y_error = mean(sqrt(square(pred_Y - real_pos[:, 1])))

        mean_error = mean(sqrt(square(pred_X - real_pos[:, 0]) + square(pred_Y - real_pos[:, 1])))
        mean_std =  mean( std(pred_X) + std(pred_Y) )
        
        return  X_error, Y_error, mean_error, mean_std

    def pseudolabels(self, x, batch_size):

        # Convert Input
        x = Utilities.normalizeX(data=x, size=self.features, b=self.b)

        # Load model
        model = load_model(self.model_path)

        # Derive Prediction
        predict_pos = model.predict(x, batch_size=self.batch_size)

        # Normalize Output
        predict_X, predict_Y = self.normY.reverse_normalizeY(predict_pos[:, 0],predict_pos[:, 1])

        # Zip to prediction and convert to proper format
        pseudo_labels = array(list(zip(predict_X, predict_Y)))
        pseudo_labels = reshape(pseudo_labels,(-1,2))
        return pseudo_labels
