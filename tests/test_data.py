import os
import torch
from tests import _PATH_DATA

N_SAMPLES = 10222
IMAGE_SHAPE = (3, 224, 224)
LABELS_RANGE = (0, 119)

class TestData:
    def load_datasets(self):
        processed_data_path = os.path.join(_PATH_DATA, 'processed')
        train_dataset = torch.load(os.path.join(processed_data_path, 'train_data.pt'))
        val_dataset = torch.load(os.path.join(processed_data_path, 'val_data.pt'))
        test_dataset = torch.load(os.path.join(processed_data_path, 'test_data.pt'))
        return train_dataset, val_dataset, test_dataset

    def test_num_samples(self):
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        num_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert num_samples == N_SAMPLES

    def test_shapes(self):
        for dataset in self.load_datasets():
            for image, label in dataset:
                assert image.shape == IMAGE_SHAPE
                assert label.shape == torch.Size([])

    def test_labels_range(self):
        for dataset in self.load_datasets():
            for _, label in dataset:
                assert label >= LABELS_RANGE[0]
                assert label <= LABELS_RANGE[1]

    def test_normalization(self):
        for dataset in self.load_datasets():
            mean = torch.mean(dataset.tensors[0])
            std = torch.std(dataset.tensors[0])
            assert torch.isclose(mean, torch.tensor(0.), atol=1e-2)
            assert torch.isclose(std, torch.tensor(1.), atol=1e-2)
