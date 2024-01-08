from timm import create_model
import torch
from torch import nn
from torch.utils.data import TensorDataset

# TODO logging
# TODO check pep8

NUM_CLASSES = 120
BATCH_SIZE = 32 # TODO config
LEARNING_RATE = 0.001 # TODO config
EPOCHS = 10 # TODO config

def train(model, train_loader, epochs, loss_fn, optimizer, device):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}')

if __name__ == '__main__':
    images_train = torch.load('data/processed/images_train.pt')
    labels_train = torch.load('data/processed/labels_train.pt')

    train_dataset = TensorDataset(images_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('resnet50', num_classes=NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, EPOCHS, loss_fn, optimizer, device)
