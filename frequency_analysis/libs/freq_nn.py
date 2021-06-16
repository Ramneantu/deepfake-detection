import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl


class DeepFreq(pl.LightningModule):
    def __init__(self, h_params, parameters_in: int = 1400, parameters_out: int = 2, n_hidden: int = 2):
        super().__init__()
        self.h_params = h_params
        self.save_hyperparameters(h_params)
        self.parameters_in = parameters_in
        self.parameters_out = parameters_out
        self.n_hidden = n_hidden
        # Layer of model
        self.scaling_layer = nn.Parameter(torch.zeros(parameters_in))
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





