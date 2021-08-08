from frequencyAnalysis.libs.FrequencySolver import FrequencySolver
from frequencyAnalysis.libs.commons import get_feature_vector
from frequencyAnalysis.libs.FreqDataset import FreqDataset
from absl import app, flags, logging
from absl.flags import FLAGS
from frequencyAnalysis.libs.freq_nn_xray import DeepFreq
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
    device = "cpu"
    if model.type == "nn":
        model.classifier = DeepFreq(model.h_params, model.parameters_in, model.parameters_out, model.n_hidden)
        model.classifier.load_state_dict(model.classifier_state_dict)
        model.classifier.to(device)
        model.classifier.eval()
    pkl_file.close()
    return model


def main():
    model_path = "/home/deepfake/emre/repo/proj-4/Models/reruns-2"
    data_path = "/home/deepfake/emre/repo/proj-4/cross-testing/saved_features"
    logging.basicConfig(filename='./experiments.log', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('-------------------- Run Cross Tests --------------------')

    for m in os.listdir(model_path):
        print("-------------------- Starting " + m +  "--------------------'")
        model = load_model(os.path.join(model_path, m))
        for d in os.listdir(data_path):
            pkl_file = open(os.path.join(data_path, d), 'rb')
            data = pickle.load(pkl_file)
            pkl_file.close()
            if model.type == "svm":
                acc = model.classifier.score(data["data"],data["label"])
                logging.info("Model: " + m + ", Data: " + d + ": " + str(acc))
            else:
                testdata = FreqDataset(data["data"].astype(np.float32), data["label"].astype(np.longlong))
                testloader = torch.utils.data.DataLoader(testdata, batch_size=8,
                                                         shuffle=False, num_workers=2)
                correct = 0
                total = 0
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                device = "cpu"
                with torch.no_grad():
                    for dat in testloader:
                        images, labels = dat
                        outputs = model.classifier.forward(images.to(device))

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted.to("cpu") == labels).sum().item()
                        if total % 50 == 0:
                            print("processed: " + str(total))

                logging.info("Model: " + m + ", Data: " + d + ": " + str(correct / total))
