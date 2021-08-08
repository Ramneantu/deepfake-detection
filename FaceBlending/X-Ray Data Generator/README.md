**DeepFakeMask.py** 

taken from the xray github module. See XrayDemo for different masking effects. We simply used the face_hull approach. Not that interesting overall


**landmarkDBGenerator** 

module that given a path to an image directory will create a json file containing all facial landmarks for the images. This is later used for finding the nearest landmarks given an image for the blending part.

Requires: (see top of file)
- trained_pred_path: pre-trained weights for dlib.shape_predictor. see: https://github.com/davisking/dlib-models
- landmark_paths: directory where to store json with landmarks (must exist before)
- img_dir_path: path to existing images for which the landmarks need to be computed

**dataSetGenerator**

Requires: (see top of file)
- landmark_path = path to precomputed landmarks
- img_dir_path = path to existing images (pristine) from which the dataset is generated (fakes and reals)
- data_set_path = path to store the dataset. Needs to exist before.

Having a directory with images, precomputed landmarks, and a directory where the new dataset should be stored, this module creates a dataset consisting of approx. 50% fakes and 50% reals

**utils**

some small helpers (data type converter, nearest_landmark_search, etc.)
