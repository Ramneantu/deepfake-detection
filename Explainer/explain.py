"""
This pipeline will run the Lime Explainer for all models and datasets we have.

We simply need to have a dataset in the form:
DataSetName
|-- real
|-- fake

real and fake dirs must exist before and should contain images
that we want to explain

After running the lime explainer on the to be explained datasets,
the folder structure will change as follows:

DataSetName
|-- real
|-- fake
|- Model1_Type_TrainData1_DFAlgo1
|-- True_Real
|-- False_Real
|-- True_Fake
|-- False_Fake
|- Model2_Type_TrainData2_DFAlgo2
|-- True_Real
|-- False_Real
|-- True_Fake
|-- False_Fake
|- Model1_Type_TrainData2_DFAlgo1
|-- True_Real
|-- False_Real
|-- True_Fake
|-- False_Fake
...
where: Type: featureExtraction+linear, Full, FineTuning, etc.

The dirs:
|-- True_Real
|-- False_Real
|-- True_Fake
|-- False_Fake
contain images explained by lime


Example: let's say we have 2 models: Xception_trained_on_c0 and FreqNet_trained_on_c0
we Have 2 to be explained datasets: c0

The structure needed before running:

c0_explain
|-- real
|-- fake

c23_explain
|-- real
|-- fake

After running this script for matching training and explaining datasets:

c0_explain
|-- real
|-- fake
|- Xception_trained_on_c0
|-- True_Real
|-- False_Real
|-- True_Fake
|-- False_Fake
|- FreqNet_trained_on_c0
|-- True_Real
|-- False_Real
|-- True_Fake
|-- False_Fake

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
        return -1
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
    os.mkdir(os.path.join(dirName, "Explainer"))
    return 0

def explain(datasetName, modelNameSpecs, batch_predict, oneIsFake, transform):
    """
    dataSetName:    String      Path to Explain_DATA_DFALGO
    modelNameSpecs: String      Model1_Type_TrainData_DFAlgo
    batch_predict:  function    predict function for batches (will be used by explainer)
    oneIsFake:      boolean     if label 1 corresponds to fake (or otherwise)
    transform:      function    transform image before explanation (e.g. resize)

    """
    a = createDatasetFolder(datasetName, modelNameSpecs)
    if a == -1:
        print("aborting " + datasetName + " " + modelNameSpecs)
        return
    explainer = lime_image.LimeImageExplainer()
    logging.basicConfig(filename=os.path.join(datasetName, modelNameSpecs, 'explainInfo.log'), format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(datasetName + " " + modelNameSpecs)
    rf = ['real','fake']
    oneFake = lambda x: 1 if x=="fake" else 0
    correct = 0
    incorrect = 0
    total = 0
    fake_ = oneFake("fake") if oneIsFake else 1-oneFake("fake")
    for kind in rf:
        print(kind)
        kind_val = oneFake(kind) if oneIsFake else 1-oneFake(kind)
        c = 1
        im_dir = os.path.join(datasetName, kind)
        im_names = list(filter(isImage, os.listdir(im_dir)))
        for filename in im_names:
            im = get_image(os.path.join(im_dir, filename))
            if transform is not None:
                im = transform(im) 
            print(str(c) + " of " + str(len(im_names)))
            c += 1
            total += 1
            explanation = explainer.explain_instance(np.array(im),
                                                     batch_predict,
                                                     top_labels=2,
                                                     hide_color=0,
                                                     num_samples=1500,
                                                     batch_size=32)
            temp, mask = explanation.get_image_and_mask(fake_, positive_only=False, num_features=15,
                                                        hide_rest=False)
            img_boundary = mark_boundaries(temp / 255.0, mask)
            explainedClass = explanation.top_labels[0]
            with open(os.path.join(datasetName, modelNameSpecs, "Explainer", filename.split('.')[0]) + '.pkl', "wb") as f:
                pickle.dump(explanation, f)

            _, pos_mask = explanation.get_image_and_mask(explainedClass, positive_only=True, num_features=15)

            classifiedAs = ""
            #p = batch_predict([np.array(im)])
            #logging.info(filename + ": " + str(p))

            if kind_val == explainedClass:
                classifiedAs = str(kind_val == explainedClass) + kind.capitalize()
                correct += 1
            else:
                classifiedAs = str(kind_val == explainedClass) + ("Real" if kind == "fake" else "Fake")
                incorrect += 1

            with open(os.path.join(datasetName, modelNameSpecs, "Masks", filename.split('.')[0]) + '.pkl', "wb") as f:
                pickle.dump(pos_mask, f)
            plt.imsave(os.path.join(datasetName, modelNameSpecs, classifiedAs, filename), img_boundary)
    logging.info("Accuracy = " + str(correct/total))


def explainExistingModels(load_model, model_batch_predict, oneIsFake=False, transform=None):
    """
    load_model      function        given a model, it needs to be initialized.
                                    This function is passed here
                                    For Xception, it would be the function to init
                                    the NN weights
    model_batch_predict function    predictor function given a model
    oneIsFake:      boolean     if label 1 corresponds to fake (or otherwise)
    transform:      function    transform image before explanation (e.g. resize)
                    
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # tell where to look for the model
    p.add_argument('--model_path', '-mp', type=str, default=None)

    # specify which data needs to be explained
    p.add_argument('--explain_data_path', '-ed', type=str, default=None)

    args = p.parse_args()

    # not used.
    # explain all model and explain_data pairs
    if args.model_path is None or args.explain_data_path is None:
        print("please provide sensible paths by using -ed and -mp."
        + " See Read Me file for more information")

    else:
        # take explain data from given data path
        model = load_model(args.model_path)
        modelName = os.path.basename(args.model_path).split(".")[0]
        explain(args.explain_data_path,
                modelName,
                model_batch_predict(model),
                oneIsFake,
                transform)
