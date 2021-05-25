import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import griddata


class FrequencySolver:
    def __init__(
            self, num_iter: int = 500, features: int = 300, epsilon: float = 1e-8
    ):
        """

        :param num_iter: number of images that should be processed
        :param features: number of features for training
        :param epsilon: small value added at weights computation for avoiding numerical issues
        """
        self.num_iter = num_iter
        self.features = features
        self.epsilon = epsilon
        self.saved_train_data = None
        self.saved_test_data = None
        self.data = {}
        self.y = []
        self.statistics = []

    def __call__(self, compute_train: bool = False, compute_test: bool = True, path_train=None, path_test=None,
                 saved_train_data=None, saved_test_data=None, crop: bool = True, **kwargs):
        if compute_train is True and path_train is None:
            raise Exception('No data path given for train data')
        if compute_test is True and path_test is None:
            raise Exception('No data path given for test data')

        if compute_train is False and saved_train_data is None:
            raise Exception('No saved train data is given')
        if compute_test is False and saved_test_data is None:
            raise Exception('No saved test data is given')

        if compute_train is True:
            train_data, train_label = self.compute_data()
        else:
            # get data from pickle object
            pass

        if compute_test is True:
            test_data, train_label = self.compute_data()


     def compute_data(self, path, label, crop: bool = True):
        if label == 1:
            label = np.ones([self.num_iter, self.features])

        data = np.zeros([self.number_iter, self.N])
        file_num = 0

        for filename in glob.glob(path+"/*"):
            img = cv2.imread(filename, 0)

            if crop:
                h = img.shape[0] // 3
                w = img.shape[0] // 3
                img = img[h:-h,w:-w]

            psd1D = get_frequencies(img, self.epsilon)

            points = np.linspace(0, self.features, num=psd1D.size)
            xi = np.linspace(0, self.features, num=self.features)

            interpolated = griddata(points, psd1D, xi, method='cubic')
            interpolated /= interpolated[0]

            data[file_num,:] = interpolated
            file_num += 1

            if file_num == self.num_iter:
                print()
                break




        return label, data
