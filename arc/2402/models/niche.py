import torch
from torch import nn
import lightning as L


class Niche_Lightning(L.LightningModule):
    def __init__(
        self,
        loss,  # options: [MSE, MAE]
        optimizer,  # options: [Adam, SGD]
        lr,  # learning rate
    ):
        # init
        super().__init__()
        self.save_hyperparameters()

        # loss function
        self.loss_func = None
        if loss == "MSE":
            self.loss_func = nn.MSELoss()
        elif loss == "MAE":
            self.loss_func = nn.L1Loss()
        elif isinstance(loss, nn.Module):
            self.loss_func = loss
        else:
            raise NotImplementedError(f"loss function {loss} not implemented")

        # learning rate
        self.lr = lr

        # optimizer
        self.optimizer = optimizer

        # to float32
        self = self.float()

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)  # return predictions

    def compute_loss(self, batch):
        x, y = batch
        x = x.float()
        y = y.float()
        loss = self.loss_func(self(x), y)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
