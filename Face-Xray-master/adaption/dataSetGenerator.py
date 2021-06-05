import numpy as np
import os
import random
import cv2
import json
import utils as ut
from DeepFakeMask import dfl_full, facehull, components, extended
from skimage import io
from skimage import transform as sktransform
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm

# TODO:
# folder structure etc.
# and implement directory creation (right now, certain dirs have to exist
# before)

landmark_path = "/Users/emrekavak/Desktop/landmarks/landmark_db.txt"
img_dir_path = "/Users/emrekavak/Desktop/c23/images/"
data_set_path = "/Users/emrekavak/Desktop/DataSet/"


def random_get_hull(landmark, img1):
    # hull_type = random.choice([0, 1, 2, 3])
    hull_type = 3
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255


def random_erode_dilate(mask, ksize=None):
    if random.random() > 0.5:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.erode(mask, kernel, 1)/255
    else:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):

    maskIndices = np.where(mask != 0)

    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack(
        (maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(
            hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0],
                                                                               maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
        1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)

    maskIndices = np.where(mask != 0)

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


class DataSetGenerator():
    def __init__(self, landmarks_db_path=landmark_path,
                 image_path=img_dir_path, data_set_path=data_set_path):
        self.landmarks_db = self._read_landmarks_pairs(landmarks_db_path)
        self.image_names = self._get_images(image_path)
        self.image_path = image_path
        self.data_set_path = data_set_path
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.05))])

    def _get_images(self, img_dir_path):
        return [i for i in os.listdir(img_dir_path) if i.lower().endswith((".png", ".jpg", ".jpeg"))]

    def _read_landmarks_pairs(self, landmark_path):
        with open(landmark_path, 'r') as myfile:
            landmark_db_f = myfile.read()
        return json.loads(landmark_db_f)

    def get_blended_face(self, background_face_path):
        background_face = io.imread(self.image_path+background_face_path)
        background_landmark = np.array(self.landmarks_db[background_face_path])
        im_y = background_face.shape[0]
        im_x = background_face.shape[1]
        
        foreground_face_path = ut.get_nearest_face(background_face_path,
                                                   self.landmarks_db)
        foreground_face = io.imread(self.image_path+foreground_face_path)
        
        # down sample before blending
        aug_size_y = random.randint(int(im_y*0.4), im_y)
        aug_size_x = random.randint(int(im_x*0.4), im_x)
        background_landmark[:, 0] = background_landmark[:, 0] * (aug_size_y/im_y)
        background_landmark[:, 1] = background_landmark[:, 1] * (aug_size_x/im_x)
        foreground_face = sktransform.resize(foreground_face, (aug_size_y, aug_size_x), preserve_range=True).astype(np.uint8)
        background_face = sktransform.resize(background_face, (aug_size_y, aug_size_x), preserve_range=True).astype(np.uint8)
        
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
        
        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        
        # filte empty mask after deformation
        if np.sum(mask) == 0:
            raise NotImplementedError

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        
        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)
        
        # resize back to default resolution
        blended_face = sktransform.resize(blended_face, (im_y, im_x), preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask, (im_y, im_x), preserve_range=True)
        mask = mask[:, :, 0:1]
        return blended_face, mask
        
    def create_dataset(self):
        for img_name in tqdm(self.image_names):
            background_face_path = img_name
            fake = random.randint(0, 1)
            if fake:
                face_img, mask = self.get_blended_face(background_face_path)
            else:
                face_img = io.imread(self.image_path+background_face_path)
                mask = np.zeros((face_img.shape[0], face_img.shape[1], 1))

            im_y = face_img.shape[0]
            im_x = face_img.shape[1]

            # randomly downsample after BI pipeline
            if random.randint(0, 1):
                aug_size_y = random.randint(int(im_y*0.4), im_y)
                aug_size_x = random.randint(int(im_x*0.4), im_x)
                face_img = Image.fromarray(face_img)
                if random.randint(0, 1):
                    face_img = face_img.resize((aug_size_y, aug_size_x), Image.BILINEAR)
                else:
                    face_img = face_img.resize((aug_size_y, aug_size_x), Image.NEAREST)
                face_img = face_img.resize((im_y, im_x), Image.BILINEAR)
                face_img = np.array(face_img)
                
            # random jpeg compression after BI pipeline
            # if random.randint(0, 1):
            #     quality = random.randint(60, 100)
            #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            #     face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
            #     face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
            
            # removed cropping here
            
            # random flip
            if random.randint(0, 1):
                face_img = np.flip(face_img, 1)
                mask = np.flip(mask, 1)
            
            im = Image.fromarray(face_img)
            im.save(self.data_set_path
                    + ("/fake/" if fake else "/real/")
                    + img_name.split(".")[0]
                    + ("_fake" if fake else "_real")+".jpeg",
                    quality=random.randint(60, 100))


def main():
    dataSetGenerator = DataSetGenerator()
    dataSetGenerator.create_dataset()


if __name__ == "__main__":
    main()
