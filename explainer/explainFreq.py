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

# from network.models import model_selection
# from dataset.transform import xception_default_data_transforms, get_pil_transform, get_preprocess_transform
# from detect_from_video import predict_with_model, get_boundingbox

def load_model(model_path: str=None):
    pkl_file = open(model_path, 'rb')
    model = pickle.load(pkl_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model.type == "nn":
        model.classifier = DeepFreq(model.h_params, model.parameters_in, model.parameters_out, model.n_hidden)
        model.classifier.load_state_dict(model.classifier_state_dict)
        model.classifier.to(device)
        model.classifier.eval()
    pkl_file.close()
    return model


def model_batch_predict(model):
    def batch_predict(images):
        # first, call the preprocessing functions
        # for this: check compute_data in FrequencySolver

        features = np.stack(tuple(get_feature_vector(np.delete(i, [1, 2], 2).squeeze(2), model.crop, model.features, model.epsilon) for i in images), axis=0)
        if model.type == "svm":
            probs = model.classifier.predict_proba(features)
        if model.type == "nn":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                logits = model.classifier.forward(torch.from_numpy(features.astype(np.float32)).to(device))
                probs_gpu = F.softmax(logits, dim=1)
                probs = probs_gpu.detach().cpu().numpy()

        return probs
    return batch_predict


if __name__ == '__main__':
    exp.explainExistingModels(load_model, model_batch_predict, oneIsFake=True)


