from ml_ops_dog_breeds.models.model import MyNeuralNet
import torch

IMAGE_SHAPE = (3, 224, 224)
LABELS_RANGE = (0, 119)
NUM_CLASSES = 120


class TestTraining:
    """Unit tests for training-related functionality.

    This class contains a set of unit tests to validate gradient computation during training
    for the MyNeuralNet model used in the Dog Breeds classification task.

    Attributes:
        None
    """
    def load_data(self):
        """Load a pre-configured instance of the MyNeuralNet model for testing.

        Returns:
            MyNeuralNet: An instance of the MyNeuralNet model.

        """
        model = MyNeuralNet(model_type='resnet18', out_features=120, lr=0.001)
        return model

    def test_gradients(self):
        """Test gradient computation during training."""
        model = self.load_data()
        x = torch.randn(1, *IMAGE_SHAPE, requires_grad=True)
        output = model(x)
        loss = torch.sum(output)
        loss.backward()
        for param in model.parameters():
            assert not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any()
