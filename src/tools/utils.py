from pandas import read_csv, get_dummies
from numpy import copy, shape, float, reshape, argmax, round


class Utilities:

    @staticmethod
    def get_data(train_dir, valid_dir, test_dir, sep, validate=False):
        if validate:
            return Utilities.load_data_perspective(data_dir=train_dir,sep=sep),Utilities.load_data_perspective(data_dir=valid_dir,sep=sep),Utilities.load_data_perspective(data_dir=test_dir,sep=sep)
        else:
            return Utilities.load_data_perspective(data_dir=train_dir,sep=sep),(None,None),Utilities.load_data_perspective(data_dir=test_dir,sep=sep)

    @staticmethod
    def load_data_perspective(data_dir, sep):

        data_frame = read_csv(data_dir,sep=';')
        data_x = data_frame.T[:sep].T
        data_y = data_frame.T[sep:].T

        return data_x.to_numpy(), data_y.to_numpy()

    @staticmethod
    def normalizeX(data, size, b=2.71828):
        return Utilities.normalizeX_powered(data, b).reshape([-1, size])

    @staticmethod
    def normalizeX_powered(data,b):

        res = copy(data).astype(float)
        for i in range(shape(res)[0]):
            for j in range(shape(res)[1]):
                if (res[i][j] >50)|(res[i][j]==None)|(res[i][j]<-95):
                    res[i][j] = 0
                elif (res[i][j]>=0):
                    res[i][j]=1

                else :
                    res[i][j] = ((95 + res[i][j])/95.0) ** b
                    # res[i][j] = (0.01 * (110 + res[i][j])) ** 2.71828

        return res

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
