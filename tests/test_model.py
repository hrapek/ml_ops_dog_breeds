from ml_ops_dog_breeds.models.model import MyNeuralNet
import torch

IMAGE_SHAPE = (3, 224, 224)
LABELS_RANGE = (0, 119)
NUM_CLASSES = 120

class TestModel:
    def load_data(self):
        model = MyNeuralNet(model_type='resnet18', out_features=120, lr=0.001)
        return model

    def test_forward_pass(self):
        model = self.load_data()
        x = torch.randn(1, *IMAGE_SHAPE)
        try:
            _ = model(x)
        except Exception as e:
            assert False, f'Forward pass failed with error: {e}'

    def test_output_shape(self):
        model = self.load_data()
        x = torch.randn(1, *IMAGE_SHAPE)
        output = model(x)
        assert output.shape == (1, NUM_CLASSES)

    def test_initialization(self):
        model = self.load_data()
        for param in model.parameters():
            assert not torch.isnan(param).any() and not torch.isinf(param).any()
