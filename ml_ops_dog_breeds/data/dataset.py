from typing import Dict
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# TODO docstrings
# TODO check PEP8


class DogBreedDataset(Dataset):
    """Dog breed dataset used for applying transformations to the raw data and saving it."""

    def __init__(self, labels: Dict[str, str], root_dir: str, label_encoder=None, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = list(labels.keys())

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(list(labels.values()))
        else:
            self.label_encoder = label_encoder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image = Image.open(self.root_dir + image_name + '.jpg')
        label = self.label_encoder.transform([self.labels[image_name]])[0]

        if self.transform:
            image = self.transform(image)

        return image, label
