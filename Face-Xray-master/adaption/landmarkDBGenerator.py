import dlib
import os
import json
import numpy as np
import utils as ut
from tqdm import tqdm

trained_pred_path = ("/Users/emrekavak/Documents/Ethical_AI/repo/proj-4/" +
                     "Face-X-Ray-master/shape_predictor_68_face_landmarks.dat")
landmark_path = "/Users/emrekavak/Desktop/celebA-Transformed/"
img_dir_path = "/Users/emrekavak/Desktop/celebA-Transformed/"


class LandmarkDatabaseGenerator():
    def __init__(self, pretrained_pred_p=trained_pred_path,
                 landmark_path=landmark_path,
                 img_dir_path=img_dir_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(trained_pred_path)
        self.imgs = self._get_images()

    def _get_images(self):
        return [i for i in os.listdir(img_dir_path) if i.lower().endswith((".png", ".jpg", ".jpeg"))]

    def get_landmarks_and_store(self, batch_size=2000):
        landmarks_dict = {}
        for im_p in tqdm(self.imgs):
            src_im = dlib.load_rgb_image(img_dir_path+im_p)
            boxes = self.detector(src_im, 1)
            box = None
            for box in boxes:
                box = boxes.pop()
                break
            else:
                continue
            landmarks = self.predictor(src_im, box)
            landmarks_np = ut.shape_to_np(landmarks)
            landmarks_dict[im_p] = landmarks_np.tolist()

        with open(landmark_path+'landmark_db.txt', 'w') as outfile:
            json.dump(landmarks_dict, outfile)


def main():
    lmGenerator = LandmarkDatabaseGenerator()
    lmGenerator.get_landmarks_and_store()


if __name__ == "__main__":
    main()
