from typing import Dict
import csv
import torch
from torchvision import transforms
from dataset import DogBreedDataset

# TODO logging

def read_labels(filepath: str) -> Dict[str, str]:
    """Read the (image, labels) tuples from the csv file into a dictionary."""

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip the header
        labels = {row[0]: row[1] for row in reader}

    return labels


if __name__ == '__main__':
    # Get the data and process it
    labels = read_labels('data/raw/labels.csv')

    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = DogBreedDataset(labels, 'data/raw/train/', transform=transform)
    torch.save(dataset, 'data/processed/dataset.pt')
