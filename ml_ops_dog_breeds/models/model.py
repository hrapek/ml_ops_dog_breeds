import torch
import timm
from torch import nn, optim
from pytorch_lightning import LightningModule

class MyNeuralNet(LightningModule):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, out_features: int) -> None:
        super().__init__()

        self.base_model = timm.create_model("resnet18", pretrained=True)

        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, out_features)    # Final prediction fc layer
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.base_model(x)
    
    def training_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = self.criterium(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def configure_optimizer(self):
        return optim.Adam(self.parameters(), lr=1e-2)
    
    def test_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = self.criterium(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics
    
if __name__ == "__main__":
    model = MyNeuralNet(out_features=120)
    