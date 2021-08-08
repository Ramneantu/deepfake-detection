import numpy as np
import random


def shape_to_np(shape, dtype="int"):
    """
    Take facial landmarks output from dlib.shape_predictor and transform to
    numpy array
    """

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def total_euclidean_distance(a, b):
    """
    given two sets of facial landmark points, a and b, calculate the summed
    euclidean distance
    """
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b, axis=1))


def get_nearest_face(image_id, landmarkDB, subsample=2000):
    """
    image_id    str                 name of image file (should be unique)
    landmarkDB  dict(str:list)      {im_id1: im1_landmarks_as_list, ...}
    subsample   int                 instead of searching in all images, just look
                                    in subsample

    Given image name (image_id), look in landmark database (landmarkDB, json dict
    that is loaded in memory) for nearest face. If subsample is provided,
    randomly take smaller subset and then look for nearest face.
    Metric for proximity is the sum of euclidean distances of each facial landmark point
    """
    min_dist = 9999999999
    curr_min_dist = min_dist
    nearest_face_id = ""

    key_sample = random.sample(
        list(landmarkDB), k=min(subsample, len(landmarkDB)))
    for key in key_sample:
        # if same image is selected, ignore it
        if key == image_id:
            continue

        # get distance
        curr_min_dist = total_euclidean_distance(np.array(landmarkDB[key]),
                                                 np.array(landmarkDB[image_id]))
        # update min
        if curr_min_dist < min_dist:
            min_dist = curr_min_dist
            nearest_face_id = key

    return nearest_face_id
