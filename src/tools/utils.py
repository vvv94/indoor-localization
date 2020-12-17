from pandas import read_csv, get_dummies
from numpy import copy, shape, float, reshape, argmax, round, array, ndindex


class Utilities:

    @staticmethod
    def get_data(train_dir, valid_dir, test_dir, features=[11,46], validate=False):

        train_set = Utilities.load_data_perspective(data_dir=train_dir,     wifi_features=features[0],  bt_features=features[1])
        test_set = Utilities.load_data_perspective(data_dir=test_dir,       wifi_features=features[0],  bt_features=features[1])
        validate_set = Utilities.load_data_perspective(data_dir=valid_dir,  wifi_features=features[0],  bt_features=features[1]) if validate else (array([]),array([]),array([]))

        return train_set, test_set, validate_set

    @staticmethod
    def load_data_perspective(data_dir, wifi_features=11, bt_features=46):

        data = read_csv(data_dir,sep=';')

        return ((data.T[:wifi_features].T).to_numpy(),
                (data.T[wifi_features:(wifi_features+bt_features)].T).to_numpy(),
                (data.T[(bt_features+wifi_features):].T).to_numpy())

    @staticmethod
    def normalizeX(data, size, exponent, limit, wifi=True):

        if wifi:
            return Utilities.normalize_wifi(data=data, exponent=exponent, accepted=limit).reshape([-1, size])
        else:
            return Utilities.normalize_beacons(data=data, exponent=exponent, accepted=limit).reshape([-1, size])

    @staticmethod
    def normalize_wifi(data, exponent, accepted=88, max=40):

        rssi = copy(data).astype(float)
        for x,y in ndindex(rssi.shape):

            # Make sure data are in correct form
            assert abs(rssi[x,y]) >= max , 'RSSI Values are corrupted :' + str(rssi[x,y])

            if abs(rssi[x,y]) > accepted:
                rssi[x,y] = 0

            elif abs(rssi[x,y]) == max:
                rssi[x,y]=1

            else :
                rssi[x,y] = ((rssi[x,y] + accepted) / float(accepted)) ** exponent

        return rssi

    @staticmethod
    def normalize_beacons(data, exponent, accepted=100, max=59):

        rssi = copy(data).astype(float)
        for x,y in ndindex(rssi.shape):

            # Make sure data are in correct form
            assert abs(rssi[x,y]) >= max , 'RSSI Values are corrupted :' + str(rssi[x,y])

            if abs(rssi[x,y]) > accepted :
                rssi[x,y] = 0

            elif abs(rssi[x,y]) == max :
                rssi[x,y]=1

            else :
                rssi[x,y] = ((rssi[x,y] + accepted) / float(accepted)) ** exponent

        return rssi

    @staticmethod
    def oneHotEncode(arr):

        return get_dummies(reshape(arr, [-1])).values

    @staticmethod
    def oneHotDecode(arr):

        return argmax(round(arr), axis=1)

    @staticmethod
    def oneHotDecode_list(arrays):

        return [argmax(round(array),axis=1) for array in arrays]

class NormY(object):

    def __init__(self):

        self.long_max=None
        self.long_min=None
        self.lati_max=None
        self.lati_min=None
        self.long_scale=None
        self.lati_scale=None

    def fit(self, long, lati):

        self.long_max=max(long)
        self.long_min=min(long)
        self.lati_max=max(lati)
        self.lati_min=min(lati)
        self.long_scale=self.long_max-self.long_min
        self.lati_scale=self.lati_max-self.lati_min

    def normalizeY(self,long_data, lati_data):

        long_data = reshape(long_data, [-1, 1])
        lati_data = reshape(lati_data, [-1, 1])
        long=(long_data-self.long_min)/self.long_scale
        lati=(lati_data-self.lati_min)/self.lati_scale

        return long,lati

    def reverse_normalizeY(self,longitude_arr, latitude_arr):

        longitude_arr = reshape(longitude_arr, [-1, 1])
        latitude_arr = reshape(latitude_arr, [-1, 1])
        long=(longitude_arr*self.long_scale)+self.long_min
        lati=(latitude_arr*self.lati_scale)+self.lati_min

        return long,lati
