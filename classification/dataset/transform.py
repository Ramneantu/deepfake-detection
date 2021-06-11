"""

Author: Andreas Rössler
"""
from torchvision import transforms

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299)
    ])

    return transf


def get_preprocess_transform():
    normalize = transforms.Normalize([0.5]*3, [0.5]*3)
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf
