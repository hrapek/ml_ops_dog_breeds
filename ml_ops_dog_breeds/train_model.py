import os
from timm import create_model
import torch
from torch import nn
from torch.utils.data import TensorDataset
from omegaconf import OmegaConf
from ml_ops_dog_breeds.models.model import MyNeuralNet
from pytorch_lightning.callbacks import ModelCheckpoint
from data.make_dataset import DogBreedsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# TODO check pep8

# hydra configs
model_config = OmegaConf.load('models/config.yaml')
trainer_config = OmegaConf.load('config.yaml')

# load data
data = DogBreedsDataModule(trainer_config.hyperparameters.num_workers)

# model class
model = MyNeuralNet(model_type = model_config.hyperparameters.model_type, 
                    out_features = model_config.hyperparameters.out_features, 
                    lr = trainer_config.hyperparameters.lr)

# monitor model checkpoints
checkpoint_callback = ModelCheckpoint(dirpath='./models', monitor='val_loss', mode='min')

# train, val datasets
train_dataloader = data.train_dataloader(batch_size=trainer_config.hyperparameters.batch_size)
val_dataloader = data.val_dataloader(batch_size=trainer_config.hyperparameters.batch_size)

# trainer with wandb logger
trainer = Trainer(
    accelerator=trainer_config.hyperparameters.device, max_epochs=trainer_config.hyperparameters.n_epochs, callbacks=[checkpoint_callback], logger=WandbLogger(project='ml_ops_dog_breeds')
)

if __name__ == '__main__':
    trainer.fit(model, train_dataloader, val_dataloader)


# NUM_CLASSES = 120

# def train(model, train_loader, epochs, loss_fn, optimizer, device):
#     model.train()
#     for epoch in range(epochs):
#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             loss = loss_fn(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             print(f'Epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}')

# if __name__ == '__main__':
#     script_dir = os.path.dirname(__file__)
#     config_path = os.path.join(script_dir, 'config.yaml')
#     config = OmegaConf.load(config_path)

#     images_train = torch.load('data/processed/images_train.pt')
#     labels_train = torch.load('data/processed/labels_train.pt')

#     batch_size = config.hyperparameters.batch_size
#     epochs = config.hyperparameters.epochs
#     learning_rate = config.hyperparameters.learning_rate

#     train_dataset = TensorDataset(images_train, labels_train)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = MyNeuralNet(out_features=120)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     train(model, train_loader, epochs, loss_fn, optimizer, device)
