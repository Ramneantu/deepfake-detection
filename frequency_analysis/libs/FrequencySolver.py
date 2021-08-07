import cv2
import numpy as np

from . import commons
from .FreqDataset import FreqDataset
from .freq_nn import DeepFreq

import glob
from matplotlib import pyplot as plt
import pickle
from PIL import Image as pil_image
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from absl import app, flags, logging
from absl.flags import FLAGS

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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

    def __call__(self, reals_path=None, fakes_path=None,
                 training_features: str = None, **kwargs):
        """"
        If data is precomputed and save, load it
        If data is new, use paths to load it and process it
        :param compute_data: set to true if you want to process new data
        :param reals_path: path to real images
        :param fakes_path: path to fake images (deepfakes)
        :param saved_data: pickle file with saved data
        :param crop: set to true if while processing you wish to crop the face area
        """
        self.reals_path = reals_path
        self.fakes_path = fakes_path
        self.training_features = training_features

        # sanity checks
        if training_features is None and ((reals_path is None) or (fakes_path is None)):
            raise Exception('No data path given')

        # test commit
        def test_fun(self, number):
            pass

        # compute data or load data
        if training_features is None:
            # fake data has label 0 and real data has label 1
            print("Processing images...")
            reals_data, reals_label = self.compute_data(reals_path, label=1)
            fakes_data, fakes_label = self.compute_data(fakes_path, label=0)
            self.data["data"] = np.concatenate((reals_data, fakes_data), axis=0)
            self.data["label"] = np.concatenate((reals_label, fakes_label), axis=0)
        else:
            # if features have been precomputed, load them
            print("Loading features...")
            pkl_file = open('./data/features/' + training_features, 'rb')
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

    def compute_data(self, path, label):
        """
        Function that takes as input dataset and returns azimuthal averages of the frequency spectrum of each image as
        well as the corresponding label
        :param path: where images to be processed can be found
        :param label: 0 for fakes, 1 for true
        :return:  data : ndarray of shape num_iter x features, processed data
                  labels:  ndarray of shape num_iter, correct label for each image
        """
        print("Started processing dataset at location {}".format(path))
        print("Processed 0/{} images".format(self.num_iter))
        if label == 1:
            label = np.ones([self.num_iter])
        else:
            label = np.zeros([self.num_iter])

        data = np.zeros(([self.num_iter, 300]))
        file_num = 0

        for filename in glob.glob(path + "/*"):
            img = cv2.imread(filename, 0)

            h = img.shape[0] // 3
            w = img.shape[1] // 3
            images = [img[h:-h, w:-w]]

            # We don't do the blocks split anymore since it's not better then just cropping
            # no_splits = 1
            # for split in range(1, no_splits):
            #     blocks = commons.split_image(img, 3 * split)
            #     images = images + blocks

            frequencies = [commons.get_frequencies(img, self.epsilon) for img in images]
            interpolated_array = [commons.interpolate_features(psd1D, self.features, cnt) for (psd1D, cnt) in
                                  zip(frequencies, range(10))]
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

    def train(self, split_dataset: bool = True, test_file: str = 'dataset.pkl', iterations: int = 1):
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


            svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86)
            svclassifier_r.fit(X_train, y_train)

            SVM_r = svclassifier_r.score(X_test, y_test)

        else:
            SVM_r = 0
            for i in range(iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86, probability=True)
                svclassifier_r.fit(X_train, y_train)

                SVM_r += svclassifier_r.score(X_test, y_test)

        print("(Average) SVM_r: " + str(SVM_r / iterations))


        # Finally we save the model
        self.type = "svm"
        self.classifier = svclassifier_r
        # output_name = './data/models/pretrained_SVM_r.pkl'
        # output = open(output_name, 'wb')
        # # pickle.dump(svclassifier_r, output)
        # pickle.dump(self, output)
        # output.close()

        if FLAGS.save_results:
            f = open('./data/results.txt', 'a+')
            experiment_number = str(FLAGS.experiment_num)
            f.write("Results for experiment " + experiment_number + "\n" +
                    "(Average) SVM_r: " + str(SVM_r / iterations) + '\n' )


    def train_NN(self, with_trainset=False, testset_path=None):
        # Precomputed data is saved in self.data
        X = self.data["data"].astype(np.float32)
        y = self.data["label"].astype(np.longlong)

        # Set working device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Device used: {}".format(device))

        # Define training, validation (and test) datasets
        phases = ['train', 'val', 'test']
        if with_trainset is True:
            train_dataset, val_dataset, test_dataset = commons.dataset_split(X, y, 0.8, with_testset=True)
        else:
            train_dataset, val_dataset, _ = commons.dataset_split(X, y, 0.8)

            pkl_file = open('./data/' + testset_path, 'rb')
            testset = pickle.load(pkl_file)
            pkl_file.close()

            test_dataset = FreqDataset(testset["data"].astype(np.float32), testset["label"].astype(np.longlong))

        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        # Define some hyperparameters
        h_params = {
            "lr": 0.0001,
            "weight_decay": 0.00000001,
            "batch_size": 256
        }

        freq_logger = TensorBoardLogger(save_dir="lightning_logs")

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20
        )

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.kaiming_uniform_(m.weight)

        # Define model
        model = DeepFreq(h_params=h_params)

        model.apply(weights_init)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs/freq_net')
        writer.add_graph(model.to(device="cuda"), torch.Tensor(X[0]).cuda())
        writer.close()

        # summary(model.cuda(), (1,1400))
        # Dataloader
        dataloader_dict = {
            x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=h_params['batch_size'], shuffle=True) if x ==
            'train' else
            torch.utils.data.DataLoader(dataset_dict[x], batch_size=h_params['batch_size'], shuffle=False)
            for x in phases}

        # Trainer
        trainer = pl.Trainer(
            max_epochs=300,
            gpus=1 if str(device) == 'cuda' else None,
            callbacks=[early_stop_callback],
            logger=freq_logger,
        )

        # Train
        trainer.fit(model, dataloader_dict['train'], dataloader_dict['val'])

        # Test
        trainer.test(test_dataloaders=dataloader_dict['test'])

        self.type = "nn"
        self.classifier_state_dict = model.state_dict()
        self.parameters_in = model.parameters_in
        self.parameters_out = model.parameters_out
        self.h_params = model.h_params
        self.n_hidden = model.n_hidden
        # output_name = './data/models/pretrained_NN.pkl'
        # output = open(output_name, 'wb')
        # pickle.dump(self, output)
        # output.close()

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
            logging.warning('No specific name given to save inputs')
        output_name = './data/' + file_name
        output = open(output_name, 'wb')
        pickle.dump(self.data, output)
        output.close()

        logging.info("Data saved in {}".format(output_name))
