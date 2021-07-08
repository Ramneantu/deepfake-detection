"""
DataSetName (Explain_DATA_DFALGO)
|-- real
|-- fake
|- Model1_Type_TrainData1_DFAlgo1
|-- T_Real
|-- F_Real
|-- T_Fake
|-- F_Fake
|- Model2_Type_TrainData2_DFAlgo2
|-- T_Real
|-- F_Real
|-- T_Fake
|-- F_Fake
|- Model1_Type_TrainData2_DFAlgo1
|-- T_Real
|-- F_Real
|-- T_Fake
|-- F_Fake
...

where: Type: featureExtraction+linear, Full, FineTuning, etc.

real and fake dirs must exist before and should contain images
that we want to explain
"""

import copy
import os
import argparse
import sys
import time
import shutil
import logging
import numpy as np
from lime import lime_image
from PIL import Image
from datetime import datetime
from PIL import Image as pil_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import pickle

IMGFILES = (".png", ".jpg", ".jpeg")

def isImage(fileName):
    return fileName.lower().endswith(IMGFILES)

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with pil_image.open(f) as img:
            return img.convert('RGB')

def createDatasetFolder(datasetName, modelNameSpecs):
    """
    dataSetName:    String      Path to Explain_DATA_DFALGO
    modelNameSpecs: String      Model1_Type_TrainData_DFAlgo
    """
    dirName = os.path.join(datasetName, modelNameSpecs)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    else:
        timeNow = "before" + datetime.now().strftime("%b%d%Y%H:%M:%S")
        old_files = os.path.join(dirName, timeNow)
        dirs = list(filter(lambda x: not(x.startswith("before")),os.listdir(dirName)))
        os.mkdir(old_files)
        for dir in dirs:
            shutil.move(os.path.join(dirName, dir), old_files)
    os.mkdir(os.path.join(dirName, "TrueFake"))
    os.mkdir(os.path.join(dirName, "FalseFake"))
    os.mkdir(os.path.join(dirName, "TrueReal"))
    os.mkdir(os.path.join(dirName, "FalseReal"))
    os.mkdir(os.path.join(dirName, "Masks"))

def explain(datasetName, modelNameSpecs, batch_predict, oneIsFake):
    """
    batch_predict: needs to handle preprocessing and resizing

    """
    createDatasetFolder(datasetName, modelNameSpecs)
    explainer = lime_image.LimeImageExplainer()
    logging.basicConfig(filename=os.path.join(datasetName, modelNameSpecs, 'explainInfo.log'), format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(datasetName + " " + modelNameSpecs)
    rf = ['real', 'fake']
    oneFake = lambda x: 1 if x=="fake" else 0
    for kind in rf:
        print(kind)
        kind_val = oneFake(kind) if oneIsFake else 1-oneFake(kind)
        c = 1
        im_dir = os.path.join(datasetName, kind)
        im_names = list(filter(isImage, os.listdir(im_dir)))
        for filename in im_names:
            im = get_image(os.path.join(im_dir, filename))
            print(str(c) + " of " + str(len(im_names)))
            c += 1
            explanation = explainer.explain_instance(np.array(im),
                                                        batch_predict,
                                                        top_labels=2,
                                                        hide_color=0,
                                                        num_samples=1500,
                                                        batch_size=32)
            temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=15,
                                                        hide_rest=False)
            img_boundary = mark_boundaries(temp / 255.0, mask)
            explainedClass = explanation.top_labels[0]

            _, pos_mask = explanation.get_image_and_mask(explainedClass, positive_only=True, num_features=15)

            classifiedAs = ""

            if kind_val == explainedClass:
                classifiedAs = str(kind_val == explainedClass) + kind.capitalize()
            else:
                classifiedAs = str(kind_val == explainedClass) + ("Real" if kind == "fake" else "Fake")

            with open(os.path.join(datasetName, modelNameSpecs, "Masks", filename.split('.')[0]) + '.pkl', "wb") as f:
                pickle.dump(pos_mask, f)
            plt.imsave(os.path.join(datasetName, modelNameSpecs, classifiedAs, filename), img_boundary)            


def explainExistingModels(load_model, model_batch_predict, oneIsFake=False):
    """
    all_models_path: directory of a specific architecture that may have different model/weight files
                e.g.: XceptionModels --> XceptionTrainedOnFF30k_full.pickle, XceptionTrainedOnCelebA_finetuning.pickle, ...
    explaing_data_path: 
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--all_models_path', '-amp', type=str, default=None)
    p.add_argument('--model_path', '-mp', type=str, default=None)
    p.add_argument('--explain_data_path', '-ed', type=str, default=None)
    p.add_argument('--all_explain_data_path', '-aed', type=str, default=None)

    args = p.parse_args()

    if args.all_models_path is not None:
        for m in os.listdir(args.all_models_path):
            model = load_model(os.path.join(args.all_models_path, m))
            modelName = os.path.basename(m).split(".")[0]
            if args.all_explain_data_path is not None:
                for data in os.listdir(args.all_explain_data_path):
                    explain(os.path.join(args.all_explain_data_path, data), 
                            modelName,
                            model_batch_predict(model),
                            oneIsFake)
            else:
                explain(args.explain_data_path, 
                        modelName,
                        model_batch_predict(model),
                        oneIsFake)


    else:
        # take explain data from given data path
        model = load_model(args.model_path)
        modelName = os.path.basename(args.model_path).split(".")[0]
        if args.all_explain_data_path is not None:
            for data in os.listdir(args.all_explain_data_path):
                explain(os.path.join(args.explain_data_path, data), 
                        modelName,
                        model_batch_predict(model),
                        oneIsFake)
        else:
            explain(args.explain_data_path, 
                    modelName,
                    model_batch_predict(model),
                    oneIsFake)
