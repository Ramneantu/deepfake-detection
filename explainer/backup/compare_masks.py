import os
import numpy as np
import argparse
import pickle


def compare_masks(x, y):
    union = np.logical_or(x, y)
    intersection = np.logical_and(x, y)
    perc = np.sum(intersection) / np.sum(union)
    return perc


def load_data(first_folder, second_folder):
    first_list = sorted(os.listdir(os.path.join(first_folder, 'Masks')))
    second_list = sorted(os.listdir(os.path.join(second_folder, 'Masks')))
    for i in range(len(first_list)):
        if first_list[i] != second_list[i]:
            raise ValueError("The two folders must contain the same filenames")
        with open(os.path.join(first_folder, 'Masks', first_list[i]), 'rb') as f:
            x_mask = pickle.load(f)
        with open(os.path.join(second_folder, 'Masks', second_list[i]), 'rb') as s:
            y_mask = pickle.load(s)
        iou = compare_masks(x_mask, y_mask)
        print(f"Masks {first_list[i]} overlap {iou*100:.1f}%")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--first_folder',  '-f', type=str)
    p.add_argument('--second_folder',  '-s', type=str)
    args = p.parse_args()

    load_data(args.first_folder, args.second_folder)

