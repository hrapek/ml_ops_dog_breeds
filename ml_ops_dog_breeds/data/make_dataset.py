from typing import Dict
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DogBreedDataset
from sklearn.preprocessing import LabelEncoder
import joblib

# TODO logging
# TODO check transformations

def read_labels(filepath: str) -> Dict[str, str]:
    """Read the (image, labels) tuples from the csv file into a dictionary."""

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip the header
        labels = {row[0]: row[1] for row in reader}

    return labels


if __name__ == '__main__':
    labels = read_labels('data/raw/labels.csv')

    label_encoder = LabelEncoder()
    label_encoder.fit(list(labels.values()))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = DogBreedDataset(labels, 'data/raw/train/', label_encoder=label_encoder, transform=transform)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_images = []
    all_labels = []
    for images, labels in data_loader:
        all_images.append(images)
        all_labels.append(labels)

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    torch.save(all_images, 'data/processed/images.pt')
    torch.save(all_labels, 'data/processed/labels.pt')
    joblib.dump(label_encoder, 'data/processed/label_encoder.pkl')
