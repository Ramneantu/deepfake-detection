import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl

# New idea for custom layer: inherit from nn.Module in a new class and then add it to the DeepFreq

class DeepFreq(pl.LightningModule):
    def __init__(self, h_params, parameters_in: int = 1400, parameters_out: int = 2, n_hidden: int = 2):
        super().__init__()
        self.save_hyperparameters(h_params)
        self.parameters_in = parameters_in
        self.parameters_out = parameters_out
        self.n_hidden = n_hidden
        # Layer of model
        # self.scaling_layer = nn.Parameter(torch.randn(parameters_in), requires_grad=True)
        self.scaling_layer = ScalingLayer(parameters_in)
        self.scaling_layer.cuda()
        self.FC = nn.Sequential(
            nn.Linear(in_features=parameters_in, out_features=700),
            nn.ReLU(),
            nn.Linear(in_features=700, out_features=350),
            nn.ReLU(),
            nn.Linear(in_features=350, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=70),
            nn.ReLU(),
            nn.Linear(in_features=70, out_features=40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        scaled_x = self.scaling_layer(x)
        predictions = self.FC(scaled_x)

        return predictions

    def general_step(self, batch, batch_idx, mode):
        signals, labels = batch
        prediction = self.forward(signals)

        entropy_loss = nn.CrossEntropyLoss()
        loss = entropy_loss(prediction, labels)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, ):
        loss = self.general_step(batch, batch_idx, "val")
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + 'loss'] for x in outputs]).mean

        return avg_loss

    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")

    def configure_optimizers(self):
        parameters = self.parameters()
        optimization_method = torch.optim.Adam(parameters, lr=self.hparams['lr'],
                                               weight_decay=self.hparams['weight_decay'])
        return optimization_method

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, mode='test')

        return {'test_loss': loss}


class ScalingLayer(nn.Module):
    """ Custom layer that takes 1D input and element-wise multiplies each element by a weight"""
    def __init__(self, size):
        """
        Define input .* weights
        :param size: Size of input. Output size of layer remains the same
        """
        super(ScalingLayer, self).__init__()
        self.size = size
        self.weights = nn.Parameter(torch.Tensor(1, size), requires_grad=True)

        nn.init.xavier_normal_(self.weights)

    def forward(self, x):
        return x * self.weights



