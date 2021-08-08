from torch.utils.data.dataset import Dataset
import numpy as np
import torch

class FreqDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray):
        self._data = torch.from_numpy(data)
        self._label = torch.from_numpy(label)

    def __len__(self):
        # No need to recalculate this value every time
        return self._label.shape[0]

    def __getitem__(self, index):
        return self._data[index], self._label[index]
