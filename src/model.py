import numpy

import torch
import torch.nn as nn
from torch.nn import functional as F

from diy_functions import accuracy_quiz


class Predictor(pl.LightningModule):
    def __init__(self, inputDim, hiddenDim, batchSize, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size=inputDim,
                           hidden_size=hiddenDim,
                           batch_first=True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
        self.batch_size = batchSize

    def get_loader(x_train, y_train, x_valid, y_volid, x_test, y_test):
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True)

        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                       batch_size=self.batch_size)

        valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                        batch_size=self.batch_size)

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

    def training_step(self, batch, batch_idx):
        x, y, candi = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        y_hat = y_hat.detach().cpu().numpy()
        y = y.cpu().numpy()
        candi = candi.cpu().numpy()

        train_acc, mean_rank = accuracy_quiz(y_hat, y, candi)        
        tensorboard_logs = {'train_loss': loss, 'train_acc': train_acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, candi = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        candi = candi.cpu().numpy()

        valid_acc, mean_rank = accuracy_quiz(y_hat, y, candi)       
        return {'val_loss': loss, 'val_acc': valid_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = 0
        for output in outputs:
            val_acc = output['val_acc']
            avg_acc += val_acc

        avg_acc /= len(outputs)

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc' : avg_acc}
        return {
            'avg_val_loss': avg_loss, 
            'avg_val_acc': avg_acc, 
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        x, y, candi = batch
        y_hat = self.forward(x)
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        candi = candi.cpu().numpy()

        test_acc, mean_rank = accuracy_quiz(y_hat, y, candi)
        output = {'test_acc': test_acc}
        return output

    def test_end(self, outputs):
        test_acc_mean = 0
        for output in outputs:
            test_acc = output['test_acc']
            test_acc_mean += test_acc

        test_acc_mean /= len(outputs)

        tqdm_dict = {'test_acc': test_acc_mean}
        #test_acc_meanはnumpyだから.item()はいらない．

        tensorboard_logs = {'test_acc': test_acc_mean}
        result = {'progress_bar' : tqdm_dict, 'log': tensorboard_logs}
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @pl.data_loader
    def train_dataloader(self):
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        return valid_loader

    @pl.data_loader
    def test_dataloader(self):
        return test_loader

