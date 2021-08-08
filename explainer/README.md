# Explainer

Example: Let's assume we want to explain some of the images from c0.

What we need:
- A directory for the to be explained data, let's call it c0Explain that contains subfolders real and fake with their corresponding images
- At least one model we want to explain the data with. Let's say Xception trained on c0. (by model we mean the stored model parameters)

What the script outputs: A new subdirectory in c0Explain that contains the predictions (true and false positives/negatives, and their corresponding lime explanation segments.)

Visually depicted:

![Semantic description of image](ExplainerStructure.png)


## Modules and Scripts

- explain.py: wrapper that handles the explanation procedure.
- explainFreq.py: module that calls explain.py and implements the loaders and transformations for the SVM and NN frequency analyzers.
- explainFreq.py: module that calls explain.py and implements the loaders and transformations for the XceptionNet.
- subsampleImg.py: helper that was used to subsample from given image folders


