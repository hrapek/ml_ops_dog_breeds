import csv
from PIL import Image
import joblib
import torch
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder

import os

_DATASCRIPTS_ROOT = os.path.dirname(__file__)  # root of this folder
_PROJECT_ROOT = os.path.dirname(_DATASCRIPTS_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, 'data')  # path of data (not this data folder, but global)


class DogBreedsDataModule(LightningDataModule):
    """Lightning DataModule for loading and processing Dog Breeds classification data.

    Args:
        load_path (str): Path to the raw data directory (default: 'data/raw').
        save_path (str): Path to the processed data directory (default: 'data/processed').
        num_workers (int): Number of workers for data loading (default: 1).

    Attributes:
        load_path (str): Path to the raw data directory.
        save_path (str): Path to the processed data directory.
        num_workers (int): Number of workers for data loading.

    Example:
        Initialize the DogBreedsDataModule:
        ```python
        data_module = DogBreedsDataModule(load_path='data/raw', save_path='data/processed', num_workers=4)
        ```

    """
    def __init__(self, load_path: str = os.path.join(_PATH_DATA, 'raw'), save_path: str = os.path.join(_PATH_DATA,'processed'), num_workers: int = 1) -> None:
        super().__init__()
        self.load_path = load_path
        self.save_path = save_path
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Setup method to process and save the preprocessed data.

        Args:
            stage (str): Stage of training (fit, test, or None).

        Returns:
            None

        """
        if stage == 'fit' or stage is None:
            train, val, test, label_encoder = self.process_data(self.load_path)
            torch.save(train, f'{self.save_path}/train_data.pt')
            torch.save(val, f'{self.save_path}/val_data.pt')
            torch.save(test, f'{self.save_path}/test_data.pt')
            joblib.dump(label_encoder, f'{self.save_path}/label_encoder.pkl')

    # @profile
    def process_data(self, load_path):
        """Process and split the data into train, validation, and test sets.

        Args:
            load_path (str): Path to the raw data directory.

        Returns:
            Tuple[torch.utils.data.TensorDataset]: Train, validation, and test datasets.
            LabelEncoder: Encoder for dog breed labels.

        """

        transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        labels = self.read_labels(f'{load_path}/labels.csv')
        label_encoder = LabelEncoder()
        label_encoder.fit(list(labels.values()))

        images_tensor, labels_tensor = self.read_data(labels, label_encoder, transformations)
        n_samples = images_tensor.size(0)

        # split into train and test
        train_split_ratio = 0.8
        val_split_ratio = 0.1
        split_index_train = int(train_split_ratio * n_samples)
        split_index_val = int((train_split_ratio + val_split_ratio) * n_samples)

        indices = torch.randperm(n_samples)
        images_tensor, labels_tensor = images_tensor[indices], labels_tensor[indices]

        x_train = images_tensor[:split_index_train]
        y_train = labels_tensor[:split_index_train]
        x_val = images_tensor[split_index_train:split_index_val]
        y_val = labels_tensor[split_index_train:split_index_val]
        x_test = images_tensor[split_index_val:]
        y_test = labels_tensor[split_index_val:]

        return (
            torch.utils.data.TensorDataset(self.normalize(x_train), y_train),
            torch.utils.data.TensorDataset(self.normalize(x_val), y_val),
            torch.utils.data.TensorDataset(self.normalize(x_test), y_test),
            label_encoder,
        )

    def normalize(self, x):
        mean, std = torch.mean(x), torch.std(x)
        return (x - mean) / std

    def train_dataloader(self, batch_size):
        train = torch.load(f'{self.save_path}/train_data.pt')
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, batch_size):
        val = torch.load(f'{self.save_path}/val_data.pt')
        return torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=self.num_workers
        )

    def test_dataloader(self, batch_size):
        test = torch.load(f'{self.save_path}/test_data.pt')
        return torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=self.num_workers)

    def read_labels(self, filepath: str) -> Dict[str, str]:
        """Read the (image, labels) tuples from the csv file into a dictionary."""

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            labels = {row[0]: row[1] for row in reader}

        return labels

    def read_image(self, path, transformations):
        return transformations(Image.open(path))

    def read_data(self, labels: Dict[str, str], label_encoder, transformations) -> List:
        images_folder = f'{self.load_path}/images'
        data = [
            (
                self.read_image(f'{images_folder}/{image_id}.jpg', transformations),
                label_encoder.transform([labels[image_id]])[0],
            )
            for image_id in labels.keys()
        ]
        images_tensor = torch.stack([image for image, _ in data])
        labels_tensor = torch.tensor([label for _, label in data])
        return images_tensor, labels_tensor
