"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset
Adapted from the FaceForensics++ implementation: https://github.com/ondyari/FaceForensics
"""
import os
from os.path import join
import argparse
import subprocess
import cv2
import random
import dlib
from tqdm import tqdm
from datetime import datetime

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Get dimensions a bounding box around the face using dlib
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def extract_frames(data_path, output_path, image_prefix, frame_count = 1):
    """
    Method to extract random frames. If no face is detected or the frame cannot be extracted for some other reason,
    the frame is skipped. Process stops when enough frames have been extracted or all the frames from the video have
    been attempted.
    """
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    c = 0
    t = 0
    s = set(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))))
    while c < frame_count and t < int(reader.get(cv2.CAP_PROP_FRAME_COUNT)):
        t += 1
        frame = random.choice(list(s))
        s.remove(frame)
        reader.set(1, frame)
        success, image = reader.read()
        if not success:
            continue
        height, width = image.shape[:2]
        faces = face_detector(image, 1)
        if len(faces) == 0:
            continue
        face = faces[0]
        x, y, size = get_boundingbox(face, width, height, scale=1.5)
        cropped_face = image[y:y+size, x:x+size]
        cv2.imwrite(join(output_path, '{}_{:04d}.jpg'.format(image_prefix, frame)),
                    cropped_face)
        c += 1
    reader.release()


def extract_method_videos(data_path, dataset, compression, first_video, last_video, frames):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    global face_detector
    face_detector = dlib.get_frontal_face_detector()
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    random.seed(datetime.now())
    for video in tqdm(sorted(os.listdir(videos_path))[first_video:last_video]):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       images_path, image_folder, frames)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str, help='Where the downloaded files were stored using download_faceforensics')
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all', help="One of 'original', 'Deepfakes', 'Face2Face' and 'FaceSwap'")
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0', help="One of 'c0', 'c23', 'c40'. The higher the number, the higher the compression")
    p.add_argument('--first_video', '-fv', type=int, default=0, help="Index of the first video to extract frames from")
    p.add_argument('--last_video', '-lv', type=int, default=100, help="Index of the last video to extract frames from")
    p.add_argument('--frames', '-f', type=int, default=1, help="Number of frames to extract from each video")
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))