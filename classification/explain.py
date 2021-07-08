import copy
import os
import argparse
import sys
import time
from os.path import join
import cv2
import glob
import dlib
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

from network.models import model_selection
from dataset.transform import xception_default_data_transforms, get_pil_transform, get_preprocess_transform
from detect_from_video import predict_with_model, get_boundingbox


def batch_predict(images):
    model.eval()
    preprocess_transform = get_preprocess_transform()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with pil_image.open(f) as img:
            return img.convert('RGB')

def load_model(model_path):
    model, *_ = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        for i, param in model.named_parameters():
            param.requires_grad = False
    else:
        raise ValueError('Model not found')
    return model

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--img_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--features', '-f', type=int, default=5)
    p.add_argument('--lime', '-l', action='store_true')
    args = p.parse_args()

    logging.basicConfig(filename=os.path.join(args.img_path, 'probabilities.log'), format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('-------------------- Starting new run --------------------')
    model = load_model(args.model_path)
    explainer = lime_image.LimeImageExplainer()
    pill_transf = get_pil_transform()
    rf = ['real', 'fake']
    for kind in rf:
        dir = os.path.join(args.img_path, kind)
        c = 0
        for filename in os.listdir(dir):
            img = get_image(os.path.join(dir, filename))
            c += 1
            p = batch_predict([np.array(pill_transf(img))])
            print(f'{kind} image {c}/{len(os.listdir(dir))} ({os.path.join(dir, filename)})')
            logging.info(f'{kind} {os.path.join(dir, filename)}')
            logging.info(f'real: {p[0][0] * 100:3.1f}%')
            logging.info(f'fake: {p[0][1] * 100:3.1f}%\n')

            if args.lime:
                explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                                         batch_predict,  # xception function
                                                         top_labels=2,
                                                         hide_color=0,
                                                         num_samples=1000,
                                                         batch_size=32)
                temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=args.features,
                                                            hide_rest=False)
                img_boundry2 = mark_boundaries(temp / 255.0, mask)
                text = 'fake-classified'
                if explanation.top_labels[0] == 0:
                    text = 'real-classified'
                plt.imsave(os.path.join(args.img_path, text, filename), img_boundry2)

