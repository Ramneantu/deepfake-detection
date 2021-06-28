import copy
import os
import argparse
import sys
import time
from os.path import join
import cv2
import glob
import dlib
import torch
import random
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets
import math
import numpy as np
import pathlib
import logging
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from detect_from_video import predict_with_model, get_boundingbox

LEARNING = ['finetuning', 'full']

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, folder: str, klass: int, transform=None, extension: str = "jpg"):
        self._data = pathlib.Path(root) / folder
        self.klass = klass
        self.extension = extension
        self.transform = transform
        # Only calculate once how many files are in this folder
        # Could be passed as argument if you precalculate it somehow
        # e.g. ls | wc -l on Linux
        self._length = sum(1 for entry in os.listdir(self._data))

    def __len__(self):
        # No need to recalculate this value every time
        return self._length

    def __getitem__(self, index):
        # images always follow [0, n-1], so you access them directly
        img = Image.open(self._data / "{}.{}".format(str(index), self.extension))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.klass

def test_on_images(real_path, fake_path, model_path, cuda = True):

    # Load model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    if model_path is not None:
        model = torch.load(model_path)  # , map_location=torch.device('cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        model = model.cuda()

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    total = 0
    correct = 0
    fake_pictures = os.listdir(fake_path)
    fake_sample = random.sample(fake_pictures, 50)
    for filename in fake_sample:
        img = cv2.imread(join(fake_path, filename))
        # Image size
        height, width = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = img[y:y+size, x:x+size]

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------
            if prediction == 1:
                correct += 1
            total += 1

    real_pictures = os.listdir(real_path)
    real_sample = random.sample(real_pictures, 50)
    for filename in real_sample:
        img = cv2.imread(join(real_path, filename))
        # Image size
        height, width = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = img[y:y + size, x:x + size]

            # Actual prediction using our model
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------
            if prediction == 0:
                correct += 1
            total += 1

    print("Accuracy: " + str(correct/total))

def load_model(model_path, cuda = True, full = False):
    # Load model
    model, *_ = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    device = torch.device("cuda:0" if cuda else "cpu")
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        for i, param in model.named_parameters():
            param.requires_grad = False
        print('Model found in {}'.format(model_path))
        logging.info('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
        if not full:
            model.set_trainable_up_to(False, "last_linear")
            logging.info('Feature extraction, Xception net model')
        else:
            model.set_trainable_up_to(False, None)
            logging.info('Finetuning Xception net model')
    if cuda:
        model = model.cuda()
        logging.info('Running on GPU')

    return model, device

def test_model(model, dataloaders, device):
    model.eval()  # Set model to evaluate mode

    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):

            # Get model outputs and calculate loss
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)


        # statistics
        # changed accumulation vars to free memory
        running_corrects += float(torch.sum(preds == labels.data))

    test_acc = running_corrects / len(dataloaders['test'].dataset)

    print('Test Acc: {:4f}'.format(test_acc))
    logging.info('Test Acc: {:4f}'.format(test_acc))

def initialize_dataloaders(img_path, full=False):
    batch_size = 32
    if full:
        batch_size = 8
    # Create training and validation datasets
    # image_datasets = {x: ImageDataset(os.path.join(img_path, x, ), 'real', 0, xception_default_data_transforms[x]) + ImageDataset(os.path.join(img_path, x), 'fake', 1, xception_default_data_transforms[x]) for x in
    #                   ['train', 'val', 'test']}
    image_datasets = {x: datasets.ImageFolder(os.path.join(img_path, x), xception_default_data_transforms[x]) for x in
                      ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                        ['train', 'val', 'test']}

    logging.info('Dataset: {}'.format(img_path))

    return dataloaders_dict

def setup_training(img_path, model_path, full, cuda = True):

    model, device = load_model(model_path, cuda, full)

    dataloaders_dict = initialize_dataloaders(img_path, full)

    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    optimizer = optim.Adam(params_to_update, lr=0.0002, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    return model, dataloaders_dict, criterion, optimizer, device

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in tqdm(range(num_epochs), desc=' Epoch', bar_format='{desc:6}:{percentage:3.0f}%|{bar:40}{r_bar}{bar:-30b}', file=sys.stdout, position=0):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # print("Training...")
            else:
                model.eval()   # Set model to evaluate mode
                # print("Validating...")

            running_loss_epoch = 0.0
            running_corrects_epoch = 0

            running_loss_batch = 0
            running_corrects_batch = 0
            running_count_batch = 0

            # Iterate over data.
            batch_couter = 0
            for inputs, labels in tqdm(dataloaders[phase], file=sys.stdout, bar_format='{desc:6}:{percentage:3.0f}%|{bar:40}{r_bar}{bar:-30b}', desc=' {}'.format(phase), position=1, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss
                    logits = model(inputs)
                    probs = F.softmax(logits, dim=1)
                    loss = criterion(probs, labels)

                    _, preds = torch.max(probs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # changed accumulation vars to free memory
                running_loss_epoch += float(loss.item() * inputs.size(0))
                running_corrects_epoch += float(torch.sum(torch.eq(preds, labels)))

                running_loss_batch += float(loss.item() * inputs.size(0))
                running_corrects_batch += float(torch.sum(torch.eq(preds, labels)))
                running_count_batch += len(labels)
                batch_couter += 1
                if batch_couter % 10 == 0:
                    writer.add_scalar('{} loss'.format(phase), running_loss_batch / 10, epoch * len(dataloaders[phase]) + batch_couter)
                    writer.add_scalar('{} acc'.format(phase), running_corrects_batch / running_count_batch, epoch * len(dataloaders[phase]) + batch_couter)
                    running_loss_batch = 0
                    running_corrects_batch = 0
                    running_count_batch = 0


            epoch_loss = running_loss_epoch / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects_epoch / len(dataloaders[phase].dataset)
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('Epoch {}: {}\t Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        torch.save(best_model_wts, f'../data/models/xception_e{epoch + 1}')
        # print()

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'../data/models/xception_e{num_epochs}_final')
    return model, val_acc_history

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # p.add_argument('--real_path', '-r', type=str)
    # p.add_argument('--fake_path', '-f', type=str)
    p.add_argument('--img_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--full', '-l', action='store_true')
    p.add_argument('--epochs', '-e', type=int, default=5)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    logging.basicConfig(filename='../data/experiments.log', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('-------------------- Starting new run --------------------')
    writer = SummaryWriter(os.path.join('runs', args.img_path))
    # test_on_images(**vars(args))
    if args.model_path == None:
        model, dataloaders, criterion, optimizer, device = setup_training(args.img_path, args.model_path, args.full, args.cuda)
        model, *_ = train_model(model, dataloaders, criterion, optimizer, device, args.epochs)
        test_model(model, dataloaders, device)
    else:
        model, device = load_model(args.model_path, args.cuda, args.full)
        dataloaders = initialize_dataloaders(args.img_path)
        test_model(model, dataloaders, device)
    writer.close()
