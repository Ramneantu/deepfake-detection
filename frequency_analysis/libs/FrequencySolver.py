import cv2
import numpy as np

from . import commons
from .freq_nn import DeepFreq

import glob
from matplotlib import pyplot as plt
import pickle

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from absl import app, flags, logging
from absl.flags import FLAGS

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl


class FrequencySolver:
    def __init__(
            self, num_iter: int = 500, features: int = 300, epsilon: float = 1e-8
    ):
        """
        Object used for data preprocessing and training
        :param num_iter: number of images that should be processed
        :param features: number of features for training
        :param epsilon: small value added at weights computation for avoiding numerical issues

        Other parameter:
        - saved_data: pickle object with trained dataset
        - data: the azimutahl averages computed for all images
        - mean: mean value of azimuthal average for each feature
        - std: std of azimuthal average for each feature
        where one feature represents one frequency value
        """
        self.num_iter = num_iter
        self.features = features
        self.epsilon = epsilon
        self.data = {}
        self.mean = {}
        self.std = {}

    def __call__(self, compute_data: bool = True, reals_path=None, fakes_path=None,
                 saved_data: str = None, crop: bool = True, **kwargs):
        """"
        If data is precomputed and save, load it
        If data is new, use paths to load it and process it
        :param compute_data: set to true if you want to process new data
        :param reals_path: path to real images
        :param fakes_path: path to fake images (deepfakes)
        :param saved_data: pickle file with saved data
        :param crop: set to true if while processing you wish to crop the face area
        """
        # sanity checks
        if compute_data is True and ((reals_path is None) or (fakes_path is None)):
            raise Exception('No data path given')

        if compute_data is False and saved_data is None:
            raise Exception('No saved data is given')

        # compute data or load data
        if compute_data is True:
            # fake data has label 0 and real data has label 1
            reals_data, reals_label = self.compute_data(reals_path, label=1, crop=crop)
            fakes_data, fakes_label = self.compute_data(fakes_path, label=0, crop=crop)
            self.data["data"] = np.concatenate((reals_data, fakes_data), axis=0)
            self.data["label"] = np.concatenate((reals_label, fakes_label), axis=0)
        else:
            # if features have been precomputed, load them
            pkl_file = open('./data/' + saved_data, 'rb')
            loaded_data = pickle.load(pkl_file)
            pkl_file.close()
            # load data and labels
            self.data["data"] = new_data = loaded_data["data"]
            self.data["label"] = loaded_data["label"]
            # separate real and fake data
            mid_index = new_data.shape[0] // 2
            reals_data = new_data[:mid_index, :]
            fakes_data = new_data[mid_index:, :]

        # compute mean and std, we can use it for plots
        self.mean["reals"] = np.mean(reals_data, axis=0)
        self.mean["fakes"] = np.mean(fakes_data, axis=0)

        self.std["reals"] = np.std(reals_data, axis=0)
        self.std["fakes"] = np.std(fakes_data, axis=0)

    def compute_data(self, path, label, crop: bool = True):
        """
        Function that takes as input dataset and returns azimuthal averages of the frequency spectrum of each image as
        well as the corresponding label
        :param path: where images to be processed can be found
        :param label: 0 for fakes, 1 for true
        :param crop: true if you wish to crop face area
        :return:  data : ndarray of shape num_iter x features, processed data
                  labels:  ndarray of shape num_iter, correct label for each image
        """
        print("Started processing dataset at location {}".format(path))
        print("Processed 0/{} images".format(self.num_iter))
        if label == 1:
            label = np.ones([self.num_iter])
        else:
            label = np.zeros([self.num_iter])

        # data = np.zeros([self.num_iter, self.features])
        data = np.zeros(([self.num_iter, 1400]))
        file_num = 0

        for filename in glob.glob(path + "/*"):
            img = cv2.imread(filename, 0)

            if crop:
                h = img.shape[0] // 3
                w = img.shape[1] // 3
                img = img[h:-h, w:-w]

            images = [img]
            no_splits = 2
            # TODO: schimba parametrii in for loop si in method call
            for split in range(1, no_splits):
                blocks = commons.split_image(img, 3 * split)
                images = images + blocks

            frequencies = [commons.get_frequencies(img, self.epsilon) for img in images]
            # psd1D = commons.get_frequencies(img, self.epsilon)
            interpolated_array = [commons.interpolate_features(psd1D, self.features, cnt) for (psd1D, cnt) in zip(frequencies, range(10))]
            interpolated = np.hstack(interpolated_array)

            data[file_num, :] = interpolated
            file_num += 1

            if file_num == self.num_iter:
                print("Processed {}/{} images".format(file_num, self.num_iter))
                print("Finished processing dataset\n")
                break

            if file_num % 50 == 0:
                print("Processed {}/{} images".format(file_num, self.num_iter))

        return data, label

    def train(self, split_dataset: bool = True, test_file: str = 'dataset.pkl', iterations: int = 10):
        """
        This is the training function that uses shallow ml classifiers.
        :param split_dataset: If split is True, the function splits one dataset and uses 20% of it for testing. Else, you
        need to provide a separate test set
        :param test_file: pickle object if split_dataset is True
        :param iterations: If the dataset is split, perform iter number of iterations. In each iteration the a different test
        set is chosen.

        Output: (average) accuracy using four different classifiers
        """
        print("Training started!")
        X = self.data["data"]
        y = self.data["label"]

        if split_dataset is False:
            iterations = 1

            X_train = X
            y_train = y

            # load pickle object
            pkl_file = open('./data/' + test_file, 'rb')
            loaded_data = pickle.load(pkl_file)
            pkl_file.close()

            # load data and labels
            X_test = loaded_data["data"]
            y_test = loaded_data["label"]

            svclassifier = SVC(kernel='linear')
            svclassifier.fit(X_train, y_train)

            svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86)
            svclassifier_r.fit(X_train, y_train)

            svclassifier_p = SVC(kernel='poly', verbose=1)
            svclassifier_p.fit(X_train, y_train)

            logreg = LogisticRegression(solver='liblinear', max_iter=1000)
            logreg.fit(X_train, y_train)

            SVM = svclassifier.score(X_test, y_test)
            SVM_r = svclassifier_r.score(X_test, y_test)
            SVM_p = svclassifier_p.score(X_test, y_test)
            LR = logreg.score(X_test, y_test)

        else:
            LR = 0
            SVM = 0
            SVM_r = 0
            SVM_p = 0

            for i in range(iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                svclassifier = SVC(kernel='linear')
                svclassifier.fit(X_train, y_train)

                svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86)
                svclassifier_r.fit(X_train, y_train)

                svclassifier_p = SVC(kernel='poly')
                svclassifier_p.fit(X_train, y_train)

                logreg = LogisticRegression(solver='liblinear', max_iter=1000)
                logreg.fit(X_train, y_train)

                SVM += svclassifier.score(X_test, y_test)
                SVM_r += svclassifier_r.score(X_test, y_test)
                SVM_p += svclassifier_p.score(X_test, y_test)
                LR += logreg.score(X_test, y_test)

        print("(Average) SVM: " + str(SVM / iterations))
        print("(Average) SVM_r: " + str(SVM_r / iterations))
        print("(Average) SVM_p: " + str(SVM_p / iterations))
        print("(Average) LR: " + str(LR / iterations))

        if FLAGS.save_results:
            f = open('./data/results.txt', 'a+')
            experiment_number = str(FLAGS.experiment_num)
            f.write("Results for experiment " + experiment_number + "\n" +
                    "(Average) SVM: " + str(SVM / iterations) + '\n' +
                    "(Average) SVM_r: " + str(SVM_r / iterations) + '\n' +
                    "(Average) SVM_p: " + str(SVM_p / iterations) + '\n' +
                    "(Average) LR: " + str(LR / iterations) + '\n\n')


    def train_NN(self):
        # Precomputed data is saved in self.data
        X = self.data["data"].astype(np.float32)
        y = self.data["label"]

        # Set working device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: Implement Dataset
        # Define training, validation (and test) datasets
        phases = ['train', 'val', 'test']
        train_dataset, val_dataset, test_dataset = commons.dataset_split(X, y, 0.8)
        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        # TODO: Define hparameters
        # Define some hyperparameters
        h_params = {
            "lr": 0.001,
            "weight_decay": 0.000001,
            "batch_size": 128
            # early_stopping = True
        }

        # TODO: Define model
        model = DeepFreq(h_params)

        # TODO: Define Dataloader here / in DeepFreq
        # e.g. :  train_set = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
        # where train_dataset defined above
        dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=h_params['batch_size'], shuffle=True) for x
                            in
                            phases}

        # TODO: Define trainer + don't forget to initialize weights
        trainer = pl.Trainer(
                gpus=1,
                max_epochs=20
                  )
        # + check some other pl.Trainer arguments/parameters
        trainer.fit(model, dataloader_dict['train'], dataloader_dict['val'])

        # TODO: Test/evaluate the model
        trainer.test(test_dataloaders=dataloader_dict['test'])



    def visualize(self):
        """
        Plot the features of the real and fake images together.
        """
        x = np.arange(0, self.features, 1)
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(x, self.mean["fakes"], alpha=0.5, color='red', label='Fake', linewidth=2.0)
        ax.fill_between(x, self.mean["fakes"] - self.std["fakes"], self.mean["fakes"] + self.std["fakes"], color='red',
                        alpha=0.2)
        ax.plot(x, self.mean["reals"], alpha=0.5, color='blue', label='Real', linewidth=2.0)
        ax.fill_between(x, self.mean["reals"] - self.std["reals"], self.mean["reals"] + self.std["reals"], color='blue',
                        alpha=0.2)

        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        ax.legend(loc='best', prop={'size': 20})
        plt.xlabel("Spatial Frequency", fontsize=20)
        plt.ylabel("Power Spectrum", fontsize=20)

        if FLAGS.save_results:
            save_path = './img/experiment_' + str(FLAGS.experiment_num) + '.png'
            plt.savefig(save_path)

        plt.show()

    def save_dataset(self, file_name: str = 'dataset'):
        if file_name == 'dataset':
            logging.warning('No specific name given to save weights')
        output_name = './data/' + file_name
        output = open(output_name, 'wb')
        pickle.dump(self.data, output)
        output.close()

        logging.info("Data saved in {}".format(output_name))
