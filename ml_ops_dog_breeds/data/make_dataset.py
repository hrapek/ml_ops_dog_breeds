import os
from typing import Dict
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DogBreedDataset
from sklearn.preprocessing import LabelEncoder
import joblib
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule


# TODO: check transformations
# TODO: check pep8
# TODO: fill process_data once we make data work

class DogBreedsDataModule(LightningDataModule):
    def __init__(self, load_path: str = 'data/raw/', save_path: str = 'data/processed/', num_workers: int = 1) -> None:
        super().__init__()
        self.load_path = load_path
        self.save_path = save_path
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train, val, test = self.process_data(self.load_path)
            torch.save(train, f'{self.save_path}/train_data.pt')
            torch.save(val, f'{self.save_path}/val_data.pt')
            torch.save(test, f'{self.save_path}/test_data.pt')

    def process_data(self, load_path):
        """Return train, val and test dataloaders."""

        # load data
        # TODO: add part for loading and splitting data

        # normalize
        # TODO: add normalization

        return 'Empty function' # TODO: return torch.utils.data.TensorDataset for train, val, test

    def train_dataloader(self, batch_size):
        train = torch.load(f'{self.save_path}/train_data.pt')
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self, batch_size):
        val = torch.load(f'{self.save_path}/val_data.pt')
        return torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self, batch_size):
        test = torch.load(f'{self.save_path}/test_data.pt')
        return torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=self.num_workers)

if __name__ == '__main__':
    DogBreedsDataModule().setup()




# def read_labels(filepath: str) -> Dict[str, str]:
#     """Read the (image, labels) tuples from the csv file into a dictionary."""

#     with open(filepath, 'r') as f:
#         reader = csv.reader(f)
#         next(reader) # Skip the header
#         labels = {row[0]: row[1] for row in reader}

#     return labels

# if __name__ == '__main__':
#     script_dir = os.path.dirname(__file__)
#     config_path = os.path.join(script_dir, 'config.yaml')
#     config = OmegaConf.load(config_path)

#     labels = read_labels('data/raw/labels.csv')

#     label_encoder = LabelEncoder()
#     label_encoder.fit(list(labels.values()))

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor()
#     ])

#     dataset = DogBreedDataset(labels, 'data/raw/images/', label_encoder=label_encoder, transform=transform)

#     batch_size = config.data.batch_size
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # apply the transformations and load the data into memory
#     all_images = []
#     all_labels = []
#     for images, labels in data_loader:
#         all_images.append(images)
#         all_labels.append(labels)
#     all_images = torch.cat(all_images)
#     all_labels = torch.cat(all_labels)

#     # split into train and test
#     split_ratio = config.data.split_ratio
#     split_index = int(split_ratio * len(dataset))
#     indices = torch.randperm(len(dataset))
#     train_indices = indices[:split_index]
#     test_indices = indices[split_index:]
#     images_train = all_images[train_indices]
#     labels_train = all_labels[train_indices]
#     images_test = all_images[test_indices]
#     labels_test = all_labels[test_indices]

#     # save the data
#     torch.save(images_train, 'data/processed/images_train.pt')
#     torch.save(labels_train, 'data/processed/labels_train.pt')
#     torch.save(images_test, 'data/processed/images_test.pt')
#     torch.save(labels_test, 'data/processed/labels_test.pt')
#     joblib.dump(label_encoder, 'data/processed/label_encoder.pkl')
