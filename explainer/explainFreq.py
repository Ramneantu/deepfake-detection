from frequencyAnalysis.libs.FrequencySolver import FrequencySolver
from frequencyAnalysis.libs.commons import get_feature_vector
from absl import app, flags, logging
from absl.flags import FLAGS
from frequencyAnalysis.libs.freq_nn import DeepFreq
import explainer.explain as exp

import argparse
import os
import cv2

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as pil_image
from tqdm import tqdm
import math
import numpy as np
import pathlib
import logging
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

import pickle

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

