import os
import argparse
import random
import shutil


IMGFILES = (".png", ".jpg", ".jpeg")

def isImage(fileName):
    return fileName.lower().endswith(IMGFILES)

def subsample_images():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--path', '-p', type=str, default=None)
    p.add_argument('--numimages', '-n', type=int, default=20)

    args = p.parse_args()

    try:
        os.mkdir(args.path + "_subsample")
    except:
        print("Folder _subsample could not be created. " +
        "Please check if it's already existing.")

    im_names = list(filter(isImage, os.listdir(args.path)))
    
    imgs_samp = random.sample(im_names, args.numimages)

    for i in imgs_samp:
        shutil.copy(os.path.join(args.path, i), args.path+"_subsample")


if __name__ == '__main__':
    subsample_images()
