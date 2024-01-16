from ml_ops_dog_breeds.models.model import MyNeuralNet
import torch

IMAGE_SHAPE = (3, 224, 224)
LABELS_RANGE = (0, 119)
NUM_CLASSES = 120


class TestTraining:
    def load_data(self):
        model = MyNeuralNet(model_type='resnet18', out_features=120, lr=0.001)
        return model

    def test_gradients(self):
        model = self.load_data()
        x = torch.randn(1, *IMAGE_SHAPE, requires_grad=True)
        output = model(x)
        loss = torch.sum(output)
        loss.backward()
        for param in model.parameters():
            assert not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any()
