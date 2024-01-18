import os
import torch
import pytest
from __init__ import _PATH_DATA
from ml_ops_dog_breeds.data.dataset import DogBreedsDataModule


N_SAMPLES = 10222
IMAGE_SHAPE = (3, 224, 224)
LABELS_RANGE = (0, 119)


class TestData:
    """Unit tests for data loading and preprocessing.

    This class contains a set of unit tests to validate the data loading, shapes,
    labels range, and normalization of the datasets used in the Dog Breeds classification task.

    Attributes:
        None
    """
    def load_datasets(self):
        """Load preprocessed datasets for testing.

        Returns:
            Tuple[torch.utils.data.Dataset]: Train, validation, and test datasets.

        """
        processed_data_path = os.path.join(_PATH_DATA, 'processed')
        print(processed_data_path)
        train_dataset = torch.load(os.path.join(processed_data_path, 'train_data.pt'))
        val_dataset = torch.load(os.path.join(processed_data_path, 'val_data.pt'))
        test_dataset = torch.load(os.path.join(processed_data_path, 'test_data.pt'))
        return train_dataset, val_dataset, test_dataset

    @pytest.mark.order(1)
    def test_create_dataset(self):
        """Test the creation of the DogBreedsDataModule."""
        DogBreedsDataModule().setup()

    def test_num_samples(self):
        """Test the total number of samples in the datasets."""
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        num_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert num_samples == N_SAMPLES

    def test_shapes(self):
        """Test the shapes of images and labels in the datasets."""
        for dataset in self.load_datasets():
            for image, label in dataset:
                assert image.shape == IMAGE_SHAPE
                assert label.shape == torch.Size([])

    def test_labels_range(self):
        """Test the range of labels in the datasets."""
        for dataset in self.load_datasets():
            for _, label in dataset:
                assert label >= LABELS_RANGE[0]
                assert label <= LABELS_RANGE[1]

    def test_normalization(self):
        """Test the range of labels in the datasets."""
        for dataset in self.load_datasets():
            mean = torch.mean(dataset.tensors[0])
            std = torch.std(dataset.tensors[0])
            assert torch.isclose(mean, torch.tensor(0.0), atol=1e-2)
            assert torch.isclose(std, torch.tensor(1.0), atol=1e-2)
