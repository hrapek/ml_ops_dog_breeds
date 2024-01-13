from pytorch_lightning import Trainer
from data.dataset import DogBreedsDataModule
from models.model import MyNeuralNet

# TODO: figure out how to load models, lighting saves model checkpoint but with weird names (how to make it easier?)

# data
data = DogBreedsDataModule()

# checkpoint location
checkpoint_name = 'model.pt'
checkpoint_path = f'models/{checkpoint_name}'

# model abd trainer
model = MyNeuralNet.load_from_checkpoint(checkpoint_path)  # LightningModule
trainer = Trainer()

test_dataloader = data.test_dataloader()

if __name__ == '__main__':
    trainer.test(model, test_dataloader)
