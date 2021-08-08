**Modules and Files**

- DeepFakeMask.py: taken from the xray github module. See XrayDemo for different masking effects. We simply used the face_hull approach. Not that interesting overall
- landmarkDBGenerator: module that given a path to an image directory will create a json file containing all facial landmarks for the images. This is later used for finding the nearest landmarks given an image for the blending part.
- dataSetGenerator: Having a directory with images, precomputed landmarks, and a directory where the new dataset should be stored, this module creates a dataset consisting of approx. 50% fakes and 50% reals
- utils: some small helpers (data type converter, nearest_landmark_search, etc.)
