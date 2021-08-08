"""
Trains an XceptionNet to classify deepfakes from real images.
Adapted from FaceForensics++ implementation: https://github.com/ondyari/FaceForensics
"""

import copy
import os
import argparse
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets
import logging
import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from network.utils import LRScheduler, EarlyStopping

LEARNING = ['finetuning', 'full']
DATASETS = {
    "FF-compressed" : "../data/ff-c23-100k",
    "FF-raw" : "../data/ff-c0-30k",
    "X-ray" : "../data/XRay-dataset",
    "HQ" : "../data/HQ-dataset",
    "Fraunhofer": "../data/celebA_fraunhofer"
}


def load_model(model_path, full = False):
    """
    Loads a pretrained model an freezes weights or initializes model with ImageNet weights and makes it trainable.
    If a GPU is available, model is loaded onto GPU.
    @param model_path: pretrained xception model to be loaded. If None,
                        model is initialized with ImageNet weights and layers are trainable
    @param full: if set to true, all the layer of network are trainable. If false, only last layer, rest are frozen
    @return: model and the device of the model
    """
    model, *_ = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_path is not None:
        # Loads pretrained model
        model.load_state_dict(torch.load(model_path, map_location=device))
        for i, param in model.named_parameters():
            param.requires_grad = False
        print('Model found in {}'.format(model_path))
        logging.info('Model found in {}'.format(model_path))
    else:
        # Loads ImageNet model and sets the trainable layers
        print('No model found, initializing random model.')
        if not full:
            model.set_trainable_up_to(False, "last_linear")
            logging.info('Feature extraction, Xception net model')
        else:
            model.set_trainable_up_to(False, None)
            logging.info('Finetuning Xception net model')
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info('Running on GPU')

    return model, device


def test_model(model, dataloaders, device):
    """
    Tests the model and logs the accuracy to the log file
    """
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)

        # statistic
        running_corrects += float(torch.sum(preds == labels.data))

    test_acc = running_corrects / len(dataloaders['test'].dataset)

    print('Test Acc: {:4f}'.format(test_acc))
    logging.info('Test Acc: {:4f}'.format(test_acc))


def initialize_dataloaders(img_path, batch_size):
    # Create training, test and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(img_path, x), xception_default_data_transforms[x]) for x in
                      ['train', 'val', 'test']}

    # Create dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                        ['train', 'val', 'test']}

    logging.info('Dataset: {}'.format(img_path))

    return dataloaders_dict


def setup_training(img_path, model_path, full, batch_size):
    """
    Sets up training by initializing model, dataloaders, loss function, optimizer and device
    @param full: if true, all the weights are updated, not just the weights of the last layer
    @return: (torch.nn.Module, Dict[str, torch.utils.data.Dataloader], torch.nn.Module, torch.optim.Optimizer, torch.device)
    """
    model, device = load_model(model_path, full)
    dataloaders_dict = initialize_dataloaders(img_path, batch_size)

    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    optimizer = optim.Adam(params_to_update, lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    return model, dataloaders_dict, criterion, optimizer, device


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, early_stopping=0, lr_decay=0):
    """
    Training loop for the model. Complete with early stopping and learning rate decay. Returns a trained model
    @param model: TransferModel
        (wrapper for torch.nn.Module)
    @param dataloaders: Dict[str, torch.utils.data.Dataloader]
        dataloader dictionary. Keys are 'train', 'test' and 'val'
    @param criterion: loss function
    @param optimizer: pytorch optimizer to use
    @param device: torch.device (CPU or GPU) on which the model is stored
    @param num_epochs: maximum number of epochs to run. Training may stop sooner because of early stopping
    @param early_stopping: # epochs without any improvement to the validation loss after which to stop
    @param lr_decay: # epochs without any improvement to the validation loss after which to decrease learning rate
    @return: (model: torch.nn.Module, val_acc_history: Array) Trained model and the validation accuracy for each epoch
    """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if lr_decay > 0:
        lr_scheduler = LRScheduler(optimizer, lr_decay)
    if early_stopping > 0:
        early_stopper = EarlyStopping(early_stopping)

    training_finished = False
    for epoch in tqdm(range(num_epochs), desc=' Epoch', bar_format='{desc:6}:{percentage:3.0f}%|{bar:40}{r_bar}{bar:-30b}', file=sys.stdout, position=0):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Used for epoch accuracy
            running_loss_epoch = 0.0
            running_corrects_epoch = 0

            # Use for Tensorboard log
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
                # casted accumulation vars to free memory. Gradients are not computed
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
            logging.info('Epoch {}: {}\t Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if lr_decay > 0:
                    lr_scheduler(epoch_loss)
                if early_stopping > 0:
                    early_stopper(epoch_loss)
                    if early_stopper.early_stop:
                        training_finished = True

        if training_finished:
            break

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Ran for {} epochs'.format(len(val_acc_history)))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'../data/models/xception_e{num_epochs}_final')
    return model, val_acc_history


def save_model(model, dataset, full, epochs):
    torch.save(model.state_dict(), f'../data/models/xception_{dataset}_e{epochs}_{"finetuning" if full else "feature_extraction"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', '-d', type=str, choices=list(DATASETS.keys()), default=None)
    p.add_argument('--img_path', '-ip', type=str, default=None)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--full', '-l', action='store_true')
    p.add_argument('--epochs', '-e', type=int, default=5)
    p.add_argument('--lr_decay', type=int, default=0)
    p.add_argument('--early_stopping', type=int, default=0)
    p.add_argument('--batch_size', '-bs', type=int, default=8)
    args = p.parse_args()

    logging.basicConfig(filename='../data/experiments.log', format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('-------------------- Starting new run --------------------')
    img_path = DATASETS[args.dataset] if args.dataset is not None else args.img_path
    dataset = args.dataset if args.dataset is not None else img_path.split('/')[-1]
    writer = SummaryWriter(os.path.join('../data/cnn-runs', dataset, "finetuning" if args.full else "feature_extraction", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if args.model_path == None:
        model, dataloaders, criterion, optimizer, device = setup_training(img_path, args.model_path, args.full, args.batch_size)
        model, val_history = train_model(model, dataloaders, criterion, optimizer, device, args.epochs, args.early_stopping, args.lr_decay)
        save_model(model, dataset, args.full, len(val_history))
        test_model(model, dataloaders, device)
    else:
        model, device = load_model(args.model_path, args.full)
        dataloaders = initialize_dataloaders(img_path, args.batch_size)
        test_model(model, dataloaders, device)
    writer.close()
