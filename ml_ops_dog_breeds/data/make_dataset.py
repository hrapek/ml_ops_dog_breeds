from typing import Dict
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DogBreedDataset

# TODO logging
# TODO check transformations
# TODO handle labels

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
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = DogBreedDataset(labels, 'data/raw/train/', transform=transform)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_images = []
    for images, labels in data_loader:
        all_images.append(images)

    torch.save(all_images, 'data/processed/images.pt')
