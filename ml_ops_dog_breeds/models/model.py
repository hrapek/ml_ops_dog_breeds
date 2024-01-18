import torch
import timm
from torch import nn, optim
from pytorch_lightning import LightningModule


class MyNeuralNet(LightningModule):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, model_type: str, out_features: int, lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.out_features = out_features
        self.lr = lr

        self.base_model = timm.create_model(self.model_type, pretrained=True)

        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.out_features),
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = self.base_model(x)
        return x

    def training_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = self.criterium(preds, labels.long())
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterium(preds, labels.long())
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = self.criterium(preds, labels.long())
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return optim.Adam(self.base_model.fc.parameters(), lr=self.lr)


if __name__ == '__main__':
    model = MyNeuralNet(out_features=120)
