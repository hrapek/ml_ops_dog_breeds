from pytorch_lightning import Trainer
from data.dataset import DogBreedsDataModule
from models.model import MyNeuralNet
from omegaconf import OmegaConf

# TODO: figure out how to load models, lighting saves model checkpoint but with weird names (how to make it easier?)

# data
data = DogBreedsDataModule()

# checkpoint location
checkpoint_name = 'epoch=0-step=128-v1.ckpt'
checkpoint_path = f'models/{checkpoint_name}'

# hydra configs
trainer_config = OmegaConf.load('ml_ops_dog_breeds/config.yaml')

# model abd trainer
model = MyNeuralNet.load_from_checkpoint(checkpoint_path)  # LightningModule
trainer = Trainer()

test_dataloader = data.test_dataloader(batch_size=trainer_config.hyperparameters.batch_size)

if __name__ == '__main__':
    trainer.test(model, test_dataloader)
