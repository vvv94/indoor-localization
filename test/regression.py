import pandas as pd
import numpy as np
from os import environ
from warnings import  filterwarnings
#######################################
# Settings
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
environ["CUDA_VISIBLE_DEVICES"] = ''
filterwarnings('ignore')
#######################################

from tensorflow.config.experimental import (VirtualDeviceConfiguration, list_physical_devices, set_virtual_device_configuration)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Input, Reshape, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.backend import clear_session
from sklearn import svm, preprocessing

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
import random
import tensorflow as tf

DEBUG = False
ED = True
SVM = False
NN = False
CNN = True
Compare = False
seed = 666
verbose = 2

if __name__ == '__main__':

    ############################################################################################################################################
    # Clear Previous Session and Configure GPU
    ############################################################################################################################################
    # Setup GPU
    gpus = list_physical_devices("GPU")
    if not gpus:
        print("No GPU's available. Will run on CPU.")
    else:
        set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=8192)])
    # Clear TF state
    clear_session()
    random.seed(seed)
    tf.random.set_seed(seed)
    ############################################################################################################################################

    ############################################################################################################################################
    # Pre-process Data
    ############################################################################################################################################
    # Read train data
    dataset = pd.read_csv("Test.csv",sep=';',header = 0)
    rss_values = np.asarray(dataset.iloc[:,0:-2])
    locations = np.asarray(dataset.iloc[:,-2:])
    loc = np.hsplit(locations,2)
    ############################################################################################################################################
    #Read test data
    test_dataset = pd.read_csv("Train.csv",sep=';',header = 0)
    rss_values_test = np.asarray(test_dataset.iloc[:,0:-2])
    test_locations= np.asarray(test_dataset.iloc[:,-2:])
    test_loc = np.hsplit(test_locations,2)
    ############################################################################################################################################
    # Min Search
    x_origin_train,y_origin_train,x_origin_test,y_origin_test = np.amin(loc[0],axis=0), np.amin(loc[1],axis=0), np.amin(test_loc[0],axis=0), np.amin(test_loc[1],axis=0)
    origin =  np.amin(np.array([[x_origin_train, x_origin_test], [y_origin_train, y_origin_test]]), axis=1)
    ############################################################################################################################################
    # Max Search
    x_room_size_train, y_room_size_train, x_room_size_test, y_room_size_test = np.amax(loc[0], axis=0) - x_origin_train, np.amax(loc[1], axis=0) - y_origin_train, np.amax(test_loc[0], axis=0) - x_origin_test, np.amax(test_loc[1], axis=0) - y_origin_test
    room_size =  np.squeeze(np.amax(np.array([[x_room_size_train, x_room_size_test], [y_room_size_train, y_room_size_test]]), axis=1))
    if DEBUG:
        print(x_origin_train,x_origin_test,y_origin_train,y_origin_test)
        print(x_room_size_train,x_room_size_test,y_room_size_train,y_room_size_test)
    ############################################################################################################################################
    # Scale Dataset to [0,1]
    train_val_X, train_val_Y = preprocessing.scale(np.asarray(rss_values, dtype=np.float64)), np.hstack((loc[0] - origin[0], loc[1] - origin[1]))
    train_points = len(train_val_X)
    test_X, test_Y = preprocessing.scale(np.asarray(rss_values_test, dtype=np.float64)), np.hstack((test_loc[0] - origin[0], test_loc[1] - origin[1]))
    test_points = len(test_X)
    ############################################################################################################################################
    # Find training position in the training data and permutation
    # Build Unique Training Grid
    unique_position = np.vstack({tuple(row) for row in train_val_Y})
    #
    train_val_class = np.zeros(len(train_val_Y))#create the array to store class of training data
    num_unique_position = len(unique_position) #how many points in training grid
    for i in range(num_unique_position): #for each point in training grid
        in_this_class = train_val_Y[:] == unique_position[i] #find the index which has the same position as the training grid
        in_this_class = in_this_class[:,0]
        train_val_class[in_this_class]= i #label them
        sample_in_this_class = train_val_X[in_this_class]  # training sample with same location (prepared for permutation)
    ############################################################################################################################################
    # Split Training/Validation data
    train_X, val_X, train_Y, val_Y = train_test_split(train_val_X, train_val_Y, test_size=0.2, random_state=seed)
    ############################################################################################################################################
    # Draw RF Positions : Red Train/ Blue Test
    if DEBUG:
        x_tr,y_tr = train_val_Y.T
        x_te,y_te = test_Y.T
        plt.scatter(x_tr,y_tr,color='blue')
        plt.scatter(x_te,y_te,color='red')
        plt.show()
    ############################################################################################################################################

    ############################################################################################################################################
    #Euclidean distance : Calculate the Euclidean distance of fingerprints of all test points and training grids.
    ############################################################################################################################################
    if ED:
        # Initialize Variables
        num_test, ED_knn = test_X.shape[0], 3
        # Fingerprint_distance[n][k]: n indicates which test point, k indicates which training point
        fingerprint_distance= np.zeros((num_test,len(train_val_X)))
        for i in range(num_test):
            for j in range(len(train_val_X)):
                fingerprint_distance[i][j] = np.linalg.norm(train_val_X[j]-test_X[i]) #for i in range(num_test) for j in range(len(train_val_X))]
        # Find the index of largest neighbours
        nearest_neighbour_ED = [np.argpartition(fingerprint_distance[i], ED_knn)[:ED_knn] for i in range(num_test)]
        # Error of estimation
        error_ED = [np.linalg.norm(np.mean(train_val_Y[nearest_neighbour_ED[i]], axis=0) - test_Y[i]) for i in range(num_test)]
    ############################################################################################################################################

    ############################################################################################################################################
    # Support Vector Machine
    ############################################################################################################################################
    if SVM:
        # Initialize parameters
        SVM_knn, SVM_C, SVM_gamma  = 3, 7, 'auto' #1/num_feature
        # Non-linear SVM
        SVM = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=SVM_gamma, C=SVM_C)
        # Feed rssi and labels
        SVM.fit(train_val_X, train_val_class)
        # Scores for each test point n_test * n_training
        test_score = SVM.decision_function(test_X)
        # Find the index of largest neighbours
        nearest_neighbour_SVM = [np.argpartition(test_score[i], -SVM_knn)[-SVM_knn:] for i in range(num_test)]
        # Error of estimation
        error_svm = [np.linalg.norm(np.mean(unique_position[nearest_neighbour_SVM[i]], axis=0) - test_Y[i]) for i in range(num_test)]
    ############################################################################################################################################

    ############################################################################################################################################
    # Neuron Network Regressor
    ############################################################################################################################################
    if NN:
        # Parameters
        num_input = train_X.shape[1]
        act_fun = 'relu'
        epochs = 100
        drop_rate = 0.3
        reg_penalty = 0.03
        ker_init_method = 'he_normal'
        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        ############################################################################################################################################
        # Model
        model = Sequential()
        model.add(Dense(500, activation=act_fun, input_dim=num_input, kernel_initializer=ker_init_method ,kernel_regularizer=regularizers.l2(reg_penalty)))
        model.add(Dropout(drop_rate))
        model.add(Dense(500, activation=act_fun, kernel_initializer=ker_init_method ,kernel_regularizer=regularizers.l2(reg_penalty)))
        model.add(Dropout(drop_rate))
        model.add(Dense(500, activation=act_fun, kernel_initializer=ker_init_method ,kernel_regularizer=regularizers.l2(reg_penalty)))
        model.add(Dropout(drop_rate))
        model.add(Dense(2, activation='linear', kernel_initializer=ker_init_method ,kernel_regularizer=regularizers.l2(reg_penalty)))
        model.compile(loss='mae',optimizer=adam, metrics=['accuracy'])
        ############################################################################################################################################
        # Define EarlyStopping
        earlyStopping = EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
        ############################################################################################################################################
        # Train Model
        model.fit(train_X, train_Y, epochs=epochs, batch_size=64, callbacks=[earlyStopping], validation_data=(val_X, val_Y),verbose=verbose)
        ############################################################################################################################################
        # Evaluate Model
        train_loss = model.evaluate(train_X,train_Y, batch_size=len(train_Y),verbose=0)
        val_loss = model.evaluate(val_X, val_Y, batch_size=len(val_Y),verbose=0)
        test_loss = model.evaluate(test_X, test_Y, batch_size=len(test_Y),verbose=0)
        if DEBUG:
            print("\nLoss, Accuracy for training data is", ", ".join(map(str, train_loss)))
            print("Loss, Accuracy for validation data is",", ".join(map(str, val_loss)))
            print("Loss, Accuracy for test data is", ", ".join(map(str, test_loss)))
        ############################################################################################################################################
        # Predict test points locations
        predict_Y = model.predict(test_X)
        error_NN = [ np.linalg.norm(predict_Y[i] - test_Y[i]) for i in range(num_test) ]
    ############################################################################################################################################

    ############################################################################################################################################
    # Convolutional Neuron Network Regressor
    ############################################################################################################################################
    if CNN:
        # Parameters
        num_input = train_X.shape[1]
        act_fun = 'relu'
        epochs = 100
        drop_rate = 0.4
        reg_penalty = 0.03
        ker_init_method = 'he_normal'
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        ############################################################################################################################################
        # Model
        model = Sequential()
        model.add(Input(shape=(num_input,1)))
        model.add(Dropout(drop_rate))
        model.add(Conv1D(filters=99, kernel_size=16, activation=act_fun))
        model.add(Conv1D(filters=66, kernel_size=16, activation=act_fun))
        model.add(Conv1D(filters=33, kernel_size=13, activation=act_fun))
        model.add(Flatten())
        model.add(Dense(2, activation='linear'))#, kernel_initializer=ker_init_method ,kernel_regularizer=regularizers.l2(reg_penalty)))
        model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])
        ############################################################################################################################################
        # Define EarlyStopping
        earlyStopping = EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')
        ############################################################################################################################################
        # Train Model
        model.fit(train_X, train_Y, epochs=epochs, batch_size=64, callbacks=[earlyStopping], validation_data=(val_X, val_Y),verbose=verbose)
        ############################################################################################################################################
        # Evaluate Model
        train_loss = model.evaluate(train_X,train_Y, batch_size=len(train_Y),verbose=0)
        val_loss = model.evaluate(val_X, val_Y, batch_size=len(val_Y),verbose=0)
        test_loss = model.evaluate(test_X, test_Y, batch_size=len(test_Y),verbose=0)
        if DEBUG:
            print("\nLoss, Accuracy for training data is", ", ".join(map(str, train_loss)))
            print("Loss, Accuracy for validation data is",", ".join(map(str, val_loss)))
            print("Loss, Accuracy for test data is", ", ".join(map(str, test_loss)))
        ############################################################################################################################################
        # Predict test points locations
        predict_Y = model.predict(test_X)
        error_CNN = [ np.linalg.norm(predict_Y[i] - test_Y[i]) for i in range(num_test) ]
    ############################################################################################################################################



    ############################################################################################################################################
    # Display Result
    ############################################################################################################################################
    print('\nThe room size is ', " x ".join(map(str, room_size)))
    print('Dataset had',train_points,'training points and',test_points,'test points')
    ############################################################################################################################################
    if ED:
        print('The average error using Euclidean Distance:','\t',                           np.mean(error_ED),
                                                            '\t', 'minimum error:',         np.amin(error_ED),
                                                            '\t\t\t', 'maximum error:',     np.amax(error_ED),
                                                            '\t', 'variance:',              np.var(error_ED))
    ############################################################################################################################################
    if SVM:
        print('The average error using SVM :',  '\t\t\t',                   np.mean(error_svm),
                                                '\t',   'minimum error:',   np.amin(error_svm),
                                                '\t',   'maximum error:',   np.amax(error_svm),
                                                '\t',   'variance:',        np.var(error_svm))
    ############################################################################################################################################
    if NN:
        print('The average error using NN regression :',    '\t',                       np.mean(error_NN),
                                                            '\t',   'minimum error:',   np.amin(error_NN),
                                                            '\t',   'maximum error:',   np.amax(error_NN),
                                                            '\t',   'variance:',        np.var(error_NN) )
    ############################################################################################################################################
    if CNN:
        print('The average error using NN regression :',    '\t',                       np.mean(error_CNN),
                                                            '\t',   'minimum error:',   np.amin(error_CNN),
                                                            '\t',   'maximum error:',   np.amax(error_CNN),
                                                            '\t',   'variance:',        np.var(error_CNN) )
    ############################################################################################################################################
    if Compare:
        plt.boxplot([error_ED, error_svm, error_NN ])
        plt.xticks([1, 2, 3], ['Euclidean Distance','Support Vector Machine', 'Neural Network'])#, 'Support Vector Machine'
        plt.show()
    ############################################################################################################################################
