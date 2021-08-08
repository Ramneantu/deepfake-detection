import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import explain as exp
from FrequencyAnalysis.libs.freq_nn import DeepFreq
from FrequencyAnalysis.libs.FrequencySolver import FrequencySolver
from FrequencyAnalysis.libs.commons import get_feature_vector

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


def model_batch_predict(model):
    def batch_predict(images):
        # first, call the preprocessing functions
        # for this: check compute_data in FrequencySolver

        features = np.stack(tuple(get_feature_vector(np.delete(i, [1, 2], 2).squeeze(2), model.parameters_in) for i in images), axis=0)
        if classifier_type == "svm":
            probs = model.predict_proba(features)
        if classifier_type == "nn":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                logits = model.forward(torch.from_numpy(features.astype(np.float32)).to(device))
                probs_gpu = F.softmax(logits, dim=1)
                probs = probs_gpu.detach().cpu().numpy()
        # if model.type == "svm":
        #     probs = model.classifier.predict_proba(features)
        # if model.type == "nn":
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     with torch.no_grad():
        #         logits = model.classifier.forward(torch.from_numpy(features.astype(np.float32)).to(device))
        #         probs_gpu = F.softmax(logits, dim=1)
        #         probs = probs_gpu.detach().cpu().numpy()

        return probs
    return batch_predict


if __name__ == '__main__':
    exp.explainExistingModels(load_model, model_batch_predict, oneIsFake=False)


