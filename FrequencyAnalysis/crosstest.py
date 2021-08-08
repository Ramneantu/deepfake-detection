import logging
import os
import torch
import numpy as np
import pickle

from libs.FrequencySolver import FrequencySolver
from libs.commons import get_feature_vector
from libs.FreqDataset import FreqDataset
from libs.freq_nn import DeepFreq


"""
This script will run the cross-testing procedure for the frequency classifiers.

The starting points are two directories:
- model: We need files of pretrained models in a directory (model_path)
- data: We need different datasets for cross-testing (data_path)

the data here will be precomputed feature vectors stored in pickle files.
See frequency_analysis directory for more information.

------------
Example:
model_dir
|-- freq_nn_trained_on_HQ
|-- freq_svm_trained_on_c0

data_dir
|-- Xray_test
|-- c0_test


Output: logging file with entries for accuracies as:
freq_nn_trained_on_HQ, Xray_test: 0.63
freq_nn_trained_on_HQ, c0_test: 0.51
freq_svm_trained_on_c0, Xray_test: 0.6
freq_svm_trained_on_c0, c0_test: 0.5
------------

"""

model_path = "../data/models-cross"
data_path = "../data/features-cross"

def load_model(model_path: str=None):
    """
    Load model given model_path

    Discern between the NN and SVM model before loading
    """
    extension = model_path.split('.')[-1]
    global classifier_type
    if extension == 'ckpt':
        model = DeepFreq.load_from_checkpoint(model_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        classifier_type = 'nn'
    else:
        pkl_file = open(model_path, 'rb')
        model = pickle.load(pkl_file)
        classifier_type = 'svm'
    return model


def main():
    # create log file (if not existent) in ../data
    logging.basicConfig(filename='../data/experiments.log', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('-------------------- Run Cross Tests --------------------')

    for m in os.listdir(model_path):
        print("-------------------- Starting " + m +  "--------------------'")
        model = load_model(os.path.join(model_path, m))
        for d in os.listdir(data_path):
            pkl_file = open(os.path.join(data_path, d), 'rb')
            data = pickle.load(pkl_file)
            pkl_file.close()
            # we have two frequency models.
            # Depending whether SVM or NN, treat differently
            # in both cases, we want to determine the accuracy given the
            # model and the test dataset
            if classifier_type == "svm":
                acc = model.score(data["data"],data["label"])
                logging.info("Model: " + m + ", Data: " + d + ": " + str(acc))
            else:
                testdata = FreqDataset(data["data"].astype(np.float32), data["label"].astype(np.longlong))
                testloader = torch.utils.data.DataLoader(testdata, batch_size=8,
                                                         shuffle=False, num_workers=2)
                correct = 0
                total = 0
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # device = "cpu"
                with torch.no_grad():
                    for dat in testloader:
                        images, labels = dat
                        outputs = model.forward(images.to(device))

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted.to("cpu") == labels).sum().item()
                        if total % 50 == 0:
                            print("processed: " + str(total))

                logging.info("Model: " + m + ", Data: " + d + ": " + str(correct / total))


if __name__ == '__main__':
    main()