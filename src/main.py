from os.path import join
from os import environ

# Global Settings
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from sys import exit
from pathlib import Path
from tools.utils import  Utilities
from tools.augmentation import Augmentation
from model.network import Network
from time import time
from numpy import reshape, float, asarray, vstack
from random import shuffle, Random

from tensorflow.config.experimental import (VirtualDeviceConfiguration,
                                            list_physical_devices,
                                            set_virtual_device_configuration)

import numpy as np

def main():

    # Configure GPU
    set_gpu_limits(gpu_id='0',gpu_memory=8024)

    # Configure Hyperparameters
    epochs = 10000
    drop_rate = 0.7
    activation = 'relu'
    loss = 'huber_loss' # huber_loss # mse
    metric = ['accuracy','mape', 'mae']
    verbose = 2
    
    # Hypothesis 1
    pseudolabelling = False
    pseudo_epochs = 300
    
    # Hypothesis 2
    separate = False

    # Hypothesis 3    
    augment = True
    max_dist=3.1
    min_size=3
        
    # Augment Data
    if augment:
        Augmentation(data_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Train.csv'),
                     augment_file=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Augment.csv')).data_augmentation(save_to_csv=True, max_dist=max_dist, min_size=min_size, seed=666)
    
    
    # Load Data
    train_set, test_set, validation_set, augment_set, train_scaler, test_scaler = Utilities.get_data(
        train_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Train.csv'),
        valid_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Validate.csv'),
        test_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Test.csv'),
        aug_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Augment.csv'),
        features=[11,46], validate=False, augment=augment)
    

    if augment:
        train_set = Utilities.concatinate_data(train_set, augment_set, shuffled=False, seed=666)

    # Load model
    model = Network(fig_path=join(Path(__file__).parent.absolute(),'logs/'),
                    model_path=join(Path(__file__).parent.absolute(),'logs/'),
                    epochs=epochs, batch_size=64, dropout_rate=drop_rate, loss=loss, metric=metric, _seed_=666, activation=activation,
                    train_scaler=train_scaler, test_scaler=test_scaler)

    # Train Model
    start = time()
    print('Training Started.')
    if separate:
        model.fit_separated(train_set=train_set, validation_set=validation_set, verbose=verbose, plot_training=False)
    else:
        model.fit(train_set=train_set, validation_set=validation_set, verbose=verbose, plot_training=False)
    print('Training completed in '+ str(time()-start) +' seconds.\n')

    # Pseudo-labelling Training
    if pseudolabelling:

        start = time()
        # Extract Pseudo-Labels
        pseudo_labels = model.get_pseudolabels(test_measurements=(test_set[0],test_set[1]))
        p_train_set = ( vstack([train_set[0], test_set[0]]), vstack([train_set[1], test_set[1]]), vstack([train_set[2], pseudo_labels]))
        #Random(666).shuffle(p_train_set[0]); Random(666).shuffle(p_train_set[1]); Random(666).shuffle(p_train_set[2])

        # Re-Train Model with pseudo-labels Model
        model.epochs = pseudo_epochs
        print('Pseudo Training Started.')
        model.fit(train_set=p_train_set, validation_set=validation_set, verbose=verbose, plot_training=False)
        print('Retrained completed in '+ str(time()-start) +' seconds.\n')

    # Evaluate Model
    start = time()
    print('Evaluation Started.')
    x_error, x_error_std, y_error, y_error_std, error_mean, error_std = model.error(test_measurements=(test_set[0],test_set[1]), test_labels=test_set[2], separate=separate)
    print('Evaluation Completed in ' + str(time()-start) + ' seconds.\n')

    # Print Results
    print('X Coordinate Error : '+ str(x_error)     + '\t (std : ' + str(x_error_std)  +').\n' +
          'Y Coordinate Error : '+ str(y_error)     + '\t (std : ' + str(y_error_std)  +').\n' +
          'Mean Error         : '+ str(error_mean)  + '\t (std : ' + str(error_std)    +').\n')
    

def set_gpu_limits(gpu_id, gpu_memory):

    environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = list_physical_devices("GPU")
    if not gpus:
        print("No GPU's available. Server will run on CPU.")
    else:
        try: # Limit GPU Space
            set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=gpu_memory)])
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':

    main()
    exit(0)