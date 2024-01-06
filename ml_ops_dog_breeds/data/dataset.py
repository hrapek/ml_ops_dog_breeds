from typing import Dict
from PIL import Image
from torch.utils.data import Dataset

# TODO docstrings

class DogBreedDataset(Dataset):
    """Dog breed dataset."""

    def __init__(self, labels: Dict[str, str], root_dir: str, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = list(labels.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(self.root_dir + image_name + '.jpg')
        label = self.labels[image_name]

        if self.transform:
            image = self.transform(image)

        return image, label
