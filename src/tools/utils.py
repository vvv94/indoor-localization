from pandas import read_csv, get_dummies
from numpy import copy, shape, float, reshape, argmax, round, array, ndindex, vstack, squeeze
from random import Random, shuffle
from math import e
from sklearn.preprocessing import MinMaxScaler

class Utilities:

    @staticmethod
    def concatinate_data(tuple1,tuple2, seed=666, shuffled=False):

        assert len(tuple1)==3
        assert len(tuple2)==3

        new_tuple = ( vstack([tuple1[0], tuple2[0]]), vstack([tuple1[1], tuple2[1]]), vstack([tuple1[2], tuple2[2]]))
        if shuffled:
            Random(seed).shuffle(new_tuple[0])
            Random(seed).shuffle(new_tuple[1])
            Random(seed).shuffle(new_tuple[2])

        return new_tuple

    @staticmethod
    def get_data(train_dir, valid_dir, test_dir, aug_dir, features=[11,46], validate=False, augment=False):

        train_set, train_scaler = Utilities.load_data_perspective(data_dir=train_dir,   wifi_features=features[0],  bt_features=features[1])
        test_set, test_scaler = Utilities.load_data_perspective(data_dir=test_dir,      wifi_features=features[0],  bt_features=features[1])
        validate_set, _ = Utilities.load_data_perspective(data_dir=valid_dir,  wifi_features=features[0],  bt_features=features[1]) if validate else (array([]),array([]),array([])), ([],[])
        augment_set, _ = Utilities.load_data_perspective(data_dir=aug_dir,     wifi_features=features[0],  bt_features=features[1]) if augment else (array([]),array([]),array([])), ([],[])

        augment_set = augment_set[0]
        
        return train_set, test_set, validate_set, augment_set, train_scaler, test_scaler

    @staticmethod
    def load_data_perspective(data_dir, wifi_features=11, bt_features=46):

        data = read_csv(data_dir,sep=';')
        
        x_scaler, y_scaler = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(X=data.T.iloc[-2].T.to_numpy().reshape(-1,1)); y_scaler.fit(X=data.T.iloc[-1].T.to_numpy().reshape(-1,1))

        return ((data.T[:wifi_features].T).to_numpy(),
                (data.T[wifi_features:(wifi_features+bt_features)].T).to_numpy(),
                (data.T[(bt_features+wifi_features):].T).to_numpy()), (x_scaler, y_scaler)
        
    @staticmethod
    def normalizeX(data, size, exponent, limit, wifi=True):

        if wifi:
            return Utilities.normalize_wifi(data=data, exponent=exponent, accepted=limit).reshape([-1, size])
        else:
            return Utilities.normalize_beacons(data=data, exponent=exponent, accepted=limit).reshape([-1, size])

    @staticmethod
    def normalize_wifi(data, exponent, accepted=109, max=40):

        rssi = copy(data).astype(float)
        for x,y in ndindex(rssi.shape):

            # Make sure data are in correct form
            assert abs(rssi[x,y]) >= max , 'RSSI Values are corrupted :' + str(rssi[x,y])

            if abs(rssi[x,y]) > accepted:
                rssi[x,y] = 0

            #elif abs(rssi[x,y]) == max:
            #    rssi[x,y]=1.0

            else :
                rssi[x,y] = ((rssi[x,y]+accepted) / float(accepted)) ** (exponent*e)

        return rssi

    @staticmethod
    def normalize_beacons(data, exponent, accepted=109, max=59):

        rssi = copy(data).astype(float)
        for x,y in ndindex(rssi.shape):

            # Make sure data are in correct form
            assert abs(rssi[x,y]) >= max , 'RSSI Values are corrupted :' + str(rssi[x,y])

            if abs(rssi[x,y]) > accepted :
                rssi[x,y] = 0

            #elif abs(rssi[x,y]) == max :
            #    rssi[x,y]=1.0

            else :
                rssi[x,y] = ((rssi[x,y]+accepted) / float(accepted)) ** (exponent*e)
        
        return rssi

    #  min=[1.85,1.85], max_x=[48.35, 35.95]
    @ staticmethod
    def normalize_lables(x,y,scaler):
        return (squeeze(scaler[0].transform(X=x.reshape(1, -1))), squeeze(scaler[1].transform(X=y.reshape(1, -1))))

    @staticmethod
    def denormalize_labels(x,y, scaler):
        return squeeze(scaler[0].inverse_transform(X=x.reshape(1, -1))), squeeze(scaler[1].inverse_transform(X=y.reshape(1, -1)))