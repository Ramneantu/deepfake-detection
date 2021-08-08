import numpy as np
import random


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def total_euclidean_distance(a, b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b, axis=1))


def get_nearest_face(image_id, landmarkDB, subsample=2000):

    min_dist = 9999999999
    curr_min_dist = min_dist
    nearest_face_id = ""

    key_sample = random.sample(list(landmarkDB), k=min(subsample, len(landmarkDB)))
    for key in key_sample:
        if key == image_id:
            continue

        curr_min_dist = total_euclidean_distance(np.array(landmarkDB[key]),
                                                 np.array(landmarkDB[image_id]))
        if curr_min_dist < min_dist:
            min_dist = curr_min_dist
            nearest_face_id = key

    return nearest_face_id
