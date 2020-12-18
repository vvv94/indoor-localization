from os.path import join
from os import remove
from pathlib import Path
from scipy.spatial.distance import cdist
from random import Random
import numpy as np
import random
import pandas as pd


class Augmentation():

    def __init__(self,data_dir, augment_file):

        self.data_dir = data_dir
        self.augment_file = augment_file
        self.data = pd.read_csv(data_dir,sep=';')
        self.unique_x = np.unique(np.hsplit((self.data.T[-2:].T).to_numpy(dtype=float), 2)[0].flatten())
        self.unique_y = np.unique(np.hsplit((self.data.T[-2:].T).to_numpy(dtype=float), 2)[1].flatten())

    def data_augmentation(self, save_to_csv=True, max_dist=5.0, min_size=3, seed=666):

        # Initialize Variables
        augment_data, added = [], 0
        # Clear File, except for first line
        with open(self.augment_file, 'r') as file:
            data = file.read().splitlines(True)
        with open(self.augment_file, 'w') as file:
            file.writelines(data[0])

        # Augmentation based on X values
        for i in range(len(self.unique_x)):
            a = np.asarray([self.data.iloc[row].to_numpy(dtype=float) for row in self.data.index[self.data['X Pos'] == self.unique_x[i]].tolist()], dtype=float)
            locations, measurements, clusters = self.find_clusters(a, max_dist=max_dist, min_size=min_size, axis=1)
            if clusters:
                new_rssi = self.augment_rssi_data(data=measurements, locations=locations, clusters=clusters, save_to_csv=save_to_csv, file_dir=self.augment_file, seed=seed)
                added+=clusters
                augment_data.append(new_rssi)

        # Augmentation based on Y values
        for i in range(len(self.unique_y)):
            a = np.asarray([self.data.iloc[row].to_numpy(dtype=float) for row in self.data.index[self.data['Y Pos'] == self.unique_y[i]].tolist()], dtype=float)
            locations, measurements, clusters = self.find_clusters(data=a, max_dist=max_dist, min_size=min_size, axis=2)
            if clusters:
                new_rssi = self.augment_rssi_data(data=measurements, locations=locations, clusters=clusters, save_to_csv=save_to_csv, file_dir=self.augment_file, seed=seed)
                added+=clusters
                augment_data.append(new_rssi)

        print('A total of ' +str(added) + ' point have been created!')
        return  augment_data

    # TODO: Calculate Centroids for location
    def find_clusters(self, data, max_dist, min_size, axis=1):

        # Sort per axis : X = 2, Y = 1
        data = data[data[:,-axis].argsort()]

        # Find Clusters
        clusters = [[] for i in range(len(data[:,-axis]))]
        points = [[] for i in range(len(data[:,-axis]))]
        for idx,cluster in enumerate(data[:,-axis]):
            for idx2,test in enumerate(data[:,-axis]):
                if float(abs(float(cluster)-float(test))) < max_dist:
                    clusters[idx].append(data[idx2,:])
            points[idx].append(data[idx,-2:])

        # Fill Clusters
        d = [[] for i in range(len([1 for cluster in clusters if len(cluster) >= min_size]))]
        p = [[] for i in range(len([1 for cluster in clusters if len(cluster) >= min_size]))]
        count=0
        for idx3,cluster in enumerate(clusters):
            if len(cluster) >= min_size:
                d[count] = np.asarray(cluster,dtype=float)
                p[count] = np.asarray(points[idx3],dtype=float)
                count+=1

        return np.squeeze(np.asarray(p,dtype=float)), np.squeeze(np.asarray(d,dtype=float)), (np.asarray(d,dtype=float)).shape[0]

    def augment_rssi_data(self, data, locations, clusters=0, save_to_csv=True, file_dir='', print_report=False, seed=666):

        random.seed(seed)
        # Special Case of only one cluster
        if len(data.shape) < 3:

            new_measurement = np.hstack([np.array([Random().choice(rssi) for rssi in (data.T[:-2])],dtype=int), locations])
            # Add new measurement to file
            if save_to_csv:
                with open(file_dir, mode='a', newline='\n') as f:
                    pd.DataFrame(new_measurement).T.to_csv(file_dir, sep=';',mode='a', header=None, index_label=False, index=None, line_terminator='\n')

            augmented = list(new_measurement)

        else:
            augmented = []
            for idx,(location,points) in enumerate(zip(locations, data)):

                # Perform Augmentation
                new_measurement = np.hstack([np.array([Random().choice(rssi) for rssi in (points.T[:-2])],dtype=int), location])
                augmented.append(new_measurement)

                # Add new measurement to file
                if save_to_csv:
                    with open(file_dir, mode='a', newline='\n') as f:
                        pd.DataFrame(new_measurement).T.to_csv(file_dir, sep=';',mode='a', header=None, index_label=False, index=None, line_terminator='\n')

        if print_report:
            print('Added '+ str(clusters) + ' RSSI measurements to file...')

        return augmented
