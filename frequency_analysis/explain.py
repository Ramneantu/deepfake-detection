from libs.FrequencySolver import FrequencySolver
from absl import app, flags, logging
from absl.flags import FLAGS

import os
import cv2

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as pil_image
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets
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

flags.DEFINE_string('img_path', None, 'Path to images to be explained')
flags.DEFINE_string('model_path', './models/', 'Path to foledr with models')
flags.DEFINE_string('model_name', None, 'Name of saved model')


def load_model(model_path: str=None, model_name: str=None):
    pkl_file = open(model_path + model_name, 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()
    return model


# def get_image(img_path: str=None):
#     pass
#     return None


def batch_predict(images):
    pass
    # first, call the preprocessing functions
    # for this: check compute_data in FrequencySolver

    return None


def main(_argv):
    logging.info("Start explainability")

    model = load_model(FLAGS.model_path, FLAGS.model_name)
    explainer = lime_image.LimeImageExplainer()

    rf = ['real', 'fake']
    for kind in rf:
        directory = os.path.join(FLAGS.img_path, kind)
        index = 0

        for filename in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, filename))

            explanation = explainer.explain_instance(image=img,
                                                     classifier_fn=batch_predict,
                                                     top_labels=2,
                                                     hide_color=0,
                                                     num_samples=1000,
                                                     batch_size=32)
            temp, mask = explanation.get_image_and_mask(label=0, positive_only=False, num_features=10, hide_rest=False)
            img_boundry = mark_boundaries(temp / 255.0, mask)

            text = 'fake_classified' if explanation.top_labels[0] == 1 else 'real-classified'
            plt.imsave(os.path.join(FLAGS.img_path, text, filename), img_boundry)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

