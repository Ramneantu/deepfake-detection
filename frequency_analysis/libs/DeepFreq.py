import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class DeepFreq(LightningModule):
    def __init__(self, parameters_in: int = 1400, parameters_out: int = 2, n_hidden: int = 2):
        super().__init__()  # TODO: change n,h,w to something that makes sense
        self.weights = nn.Parameter(torch.Tensor(1, n, h, w))
        self.FC = nn.Sequential(
            nn.Linear(in_features=parameters_in, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=2),
            nn.Softmax()
        )

    def forward(self, x):
        scaled_x = x * self.scaling_layer
        predictions = self.FC(scaled_x)

        return predictions

    # def training_step(self, batch, batch_idx):
    #     # mandatory function
    #     # return loss

    # + we need to add some other methods as well, like config_optimizer, val_step, val_end, training_end, do we want tensorboard?

    