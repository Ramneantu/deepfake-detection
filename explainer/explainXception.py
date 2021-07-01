import sys
import os
import argparse
import explainer.explain as exp
import torch

import torch.nn as nn
import torch.nn.functional as F

from XceptionDetector.classification.network.models import model_selection
from XceptionDetector.classification.dataset.transform import xception_default_data_transforms, get_pil_transform, get_preprocess_transform

MODEL_PATH = ""
EXPLAIN_DATA_PATH = ""


def model_batch_predict(model):
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
    return batch_predict

def load_model(model_path):
    model, *_ = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is not None:
        state_dict = torch.load(
            model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        for i, param in model.named_parameters():
            param.requires_grad = False
    else:
        raise ValueError('Model not found')
    return model

def load_model_from_object(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    for i, param in model.named_parameters():
            param.requires_grad = False
    return model


if __name__ == '__main__':
    exp.explainExistingModels(load_model, model_batch_predict, oneIsFake=False)
