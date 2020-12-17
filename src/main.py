from os.path import join
from os import environ

# Global Settings
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from sys import exit
from pathlib import Path
from tools.utils import  Utilities
from model.network import Network
from time import time
from numpy import concatenate, reshape, float, asarray
from random import shuffle, Random

from tensorflow.config.experimental import (VirtualDeviceConfiguration,
                                            list_physical_devices,
                                            set_virtual_device_configuration)

e =2.71828

def main():

    # Configure GPU
    set_gpu_limits(gpu_id='',gpu_memory=8024)

    # Configure Hyperparameters
    epochs = 300
    drop_rate = 0.6
    loss = 'mse'
    metric = ['accuracy','mape', 'mae']
    verbose = 0
    pseudolabelling = False
    pseudo_epochs = 1000
    separate = False

    # Wifi
    b1=1.0*e
    r1=109

    # BT
    b2=1.45*e
    r2=109

    # Load Data
    train_set, test_set, validation_set = Utilities.get_data(train_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Train.csv'),
                                                valid_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Validate.csv'),
                                                test_dir=join(Path(__file__).parent.parent.absolute(),'dataset/SoLoc/Test.csv'),
                                                features=[11,46], validate=False)

    # Load model
    model = Network(fig_path=join(Path(__file__).parent.absolute(),'logs/'),
                    model_path=join(Path(__file__).parent.absolute(),'logs/'),
                    epochs=epochs, batch_size=64, dropout_rate=drop_rate, loss=loss, metric=metric, _seed_=666,
                    wifi_b=b1, bt_b=b2)

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
        pseudo_labels = model.get_pseudolabels(test_measurements=test_set[:,0])
        p_train_set = ( Random(seed).shuffle(concatenate([train_set[:,0], test_set[:,0]])),
                        Random(seed).shuffle(concatenate([train_set[:,1], pseudo_labels])))

        # Re-Train Model with pseudo-labels Model
        model.epochs = pseudo_epochs
        print('Pseudo Training Started.')
        model.fit(train_set=p_train_set, validation_set=None, verbose=verbose, plot_training=False)
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