from os.path import join
from sys import exit
from pathlib import Path
from tools.utils import  Utilities
from model.model import EncoderDNN
from time import time
from numpy import concatenate, reshape, float, asarray
from random import shuffle, Random
import matplotlib.pyplot as plt

# Train Settings
data_features = 520 #11
epochs = 5 # 4000
drop_rate = 0.45
batch_size = 1
validate = True
optimizer = 'Adam' # Adam, SGD
loss = 'mse' # MSE, MAE, MAPE, MSLE
metric = ['accuracy','mape']

seed = 666
pseudolabelling = False
pseudo_epochs = 2000

# Monitor Parameters
verbose = 1
validate = True
plot = True
UJIndoorLoc = True

def main():

    # Load data
    if UJIndoorLoc:
        train_dir = join(Path(__file__).parent.parent.absolute(),'dataset/UJIndoorLoc/Train.csv')
        valid_dir = join(Path(__file__).parent.parent.absolute(),'dataset/UJIndoorLoc/Validate.csv')
        test_dir = join(Path(__file__).parent.parent.absolute(),'dataset/UJIndoorLoc/Test.csv')
        fig_dir = join(Path(__file__).parent.parent.absolute(),'logs/')
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = Utilities.get_data(train_dir=train_dir, valid_dir=valid_dir, test_dir=test_dir, sep=data_features, validate=validate)
        train_y, valid_y, test_y = preprocess(train_y), preprocess(valid_y), preprocess(test_y)
    else:
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = Utilities.get_data(train_dir=train_dir, valid_dir=valid_dir, test_dir=test_dir, sep=data_features, validate=validate)

    # Load Model
    model_path = join(Path(__file__).parent.absolute(),'logs/')
    model = EncoderDNN(input=data_features, model_path=model_path, epochs=epochs, dropout_rate=drop_rate, batch_size=batch_size, loss=loss, optimizer=optimizer, metric=metric, b=2.71828, _seed_=seed)

    # Train Model
    start = time()
    print('Training Started...')
    history = model.fit(x=train_x, y=train_y, val_x=valid_x, val_y=valid_y, validate=validate, verbose=verbose)
    print('Training completed in '+ str(time()-start) +' seconds.\n')

    if plot:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(fig_dir+'accuracy.png',dpi=1000)

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(fig_dir+'accuracy.png',dpi=1000)

    # Pseudo-labelling Training
    if pseudolabelling:
        start = time()
        pseudo_train_y = model.pseudolabels(x=test_x, batch_size=batch_size)
        new_train_y = concatenate([train_y, pseudo_train_y])
        new_train_x = concatenate([train_x, test_x])
        Random(seed).shuffle(new_train_y)
        Random(seed).shuffle(new_train_x)

        # Re-Train Model with pseudo-labels Model
        model.epochs = pseudo_epochs
        print('Pseudo Training Started...')
        history = model.fit(x=train_x, y=train_y, val_x=valid_x, val_y=valid_y, validate=validate, verbose=verbose)
        print('Retrained completed in '+ str(time()-start) +' seconds.\n')

        if plot:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            plt.savefig(fig_dir+'retrain_accuracy.png',dpi=1000)

    # Evaluate Model
    start = time()
    print('Evaluation Started...')
    pos_X_error, pos_Y_error, mean_error = model.error(test_input=test_x, real_pos=test_y)
    print('Evaluation Completed in ' + str(time()-start) + ' seconds.\n')
    print('X Coordinate Error : '+str(pos_X_error) +'.\nY Coordinate Error : '+str(pos_Y_error) +'.\nMean Error : '+str(mean_error) +'.\n')


def preprocess(coords):

    for i in range(len(coords)):
        for j in range(2):
            coords[i][j] = float(str(coords[i][j]).split('.')[0] + '.' + "".join(str(coords[i][j]).split('.')[1:]))

    return asarray(coords).astype('float32')

if __name__ == '__main__':

    main()
    exit(0)