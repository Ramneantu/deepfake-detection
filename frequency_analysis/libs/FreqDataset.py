from torch.utils.data.dataset import Dataset
import numpy as np

class FreqDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray):
        self._data = data
        self._label = label

    def __len__(self):
        # No need to recalculate this value every time
        return self._label.shape[0]

    def __getitem__(self, index):
        return self._data[index], self._label[index]
