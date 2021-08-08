from libs.commons import get_feature_vector
from libs.freq_nn import DeepFreq

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image as pil_image
import numpy as np
import logging
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

import pickle


def load_model(model_path: str=None):
    pkl_file = open(model_path, 'rb')
    model = pickle.load(pkl_file)
    if model.type == "nn":
        model.classifier = DeepFreq(model.h_params, model.parameters_in, model.parameters_out, model.n_hidden)
        model.classifier.load_state_dict(model.classifier_state_dict)
        model.classifier.to(device)
        model.classifier.eval()
    pkl_file.close()
    return model


def batch_predict(images):
    pass
    # first, call the preprocessing functions
    # for this: check compute_data in FrequencySolver

    features = np.stack(tuple(get_feature_vector(np.delete(i, [1, 2], 2).squeeze(2), model.crop, model.features, model.epsilon) for i in images), axis=0)
    if model.type == "svm":
        probs = model.classifier.predict_proba(features)
    if model.type == "nn":
        with torch.no_grad():
            logits = model.classifier.forward(torch.from_numpy(features.astype(np.float32)).to(device))
            probs_gpu = F.softmax(logits, dim=1)
            probs = probs_gpu.detach().cpu().numpy()

    return probs


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with pil_image.open(f) as img:
            return img.convert('L')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--img_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    args = p.parse_args()

    logging.info("Start explainability")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path)
    explainer = lime_image.LimeImageExplainer()
    # solver_object = FrequencySolver(num_iter=FLAGS.num_iter, features=FLAGS.features)

    rf = ['real', 'fake']
    for kind in rf:
        directory = os.path.join(args.img_path, kind)
        index = 0

        for filename in os.listdir(directory):
            img = get_image(os.path.join(directory, filename))

            explanation = explainer.explain_instance(image=np.array(img),
                                                     classifier_fn=batch_predict,
                                                     top_labels=2,
                                                     hide_color=0,
                                                     num_samples=1000,
                                                     batch_size=1028)
            temp, mask = explanation.get_image_and_mask(label=1, positive_only=False, num_features=10, hide_rest=False)
            img_boundry = mark_boundaries(temp / 255.0, mask)

            text = 'fake-classified' if explanation.top_labels[0] == 0 else 'real-classified'
            plt.imsave(os.path.join(args.img_path, text, filename), img_boundry)


