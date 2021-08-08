import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl


class DeepFreq(pl.LightningModule):
    def __init__(self, h_params, parameters_in: int = 300, parameters_out: int = 2, n_hidden: int = 2):
        super().__init__()
        self.save_hyperparameters(h_params)
        self.parameters_in = parameters_in
        self.parameters_out = parameters_out
        self.n_hidden = n_hidden
        self.h_params = h_params
        self.save_hyperparameters("parameters_in")
        self.save_hyperparameters("parameters_out")
        self.save_hyperparameters("n_hidden")
        self.save_hyperparameters("h_params")

        # Layers of model
        # self.scaling_layer = ScalingLayer(parameters_in)
        # self.scaling_layer.cuda()
        self.FC = nn.Sequential(
            nn.Linear(in_features=parameters_in, out_features=100),
            nn.PReLU(),
            nn.Linear(in_features=100, out_features=40),
            nn.PReLU(),
            nn.Linear(in_features=40, out_features=10),
            nn.PReLU(),
            nn.Linear(in_features=10, out_features=2),
            nn.PReLU(),
            # nn.Linear(in_features=20, out_features=2),
            # nn.PReLU(),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # scaled_x = self.scaling_layer(x)
        predictions = self.FC(x)

        return predictions

    def general_step(self, batch, batch_idx, mode):
        signals, labels = batch
        probabilities = self.forward(signals)
        # print(probabilities)
        # print(labels)


        entropy_loss = nn.CrossEntropyLoss()
        loss = entropy_loss(probabilities, labels)

        _, pred = torch.max(probabilities, 1)
        correct = torch.eq(pred, labels)

        return loss, correct

    def training_step(self, batch, batch_idx):
        loss, correct = self.general_step(batch, batch_idx, "train")
        return {'loss': loss, 'correct': correct}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tens = torch.cat([x['correct'] for x in outputs], dim=0).double()
        acc = tens.mean()
        self.log('train_loss', avg_loss)
        self.log('train_acc', acc)

    def validation_step(self, batch, batch_idx):
        loss, correct = self.general_step(batch, batch_idx, "val")
        return {'loss': loss, 'correct': correct}

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + 'loss'] for x in outputs]).mean

        return avg_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tens = torch.cat([x['correct'] for x in outputs], dim=0).double()
        acc = tens.mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        parameters = self.parameters()
        optimization_method = torch.optim.Adam(parameters, lr=self.hparams['lr'],
                                               weight_decay=self.hparams['weight_decay'])
        return optimization_method

    def test_step(self, batch, batch_idx):
        loss, correct = self.general_step(batch, batch_idx, "val")
        return {'loss': loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tens = torch.cat([x['correct'] for x in outputs], dim=0).double()
        acc = tens.mean()
        self.log('test_loss', avg_loss)
        self.log('test_acc', acc)


class ScalingLayer(nn.Module):
    """ Custom layer that takes 1D input and element-wise multiplies each element by a weight"""
    def __init__(self, size):
        """
        Define input .* weights
        :param size: Size of input. Output size of layer remains the same
        """
        super(ScalingLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, size))
        nn.init.uniform_(self.weight)

    def forward(self, x):
        # return nn.ReLU()(x * self.weight)
        return x * self.weight


