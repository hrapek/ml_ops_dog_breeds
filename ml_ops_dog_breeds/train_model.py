from omegaconf import OmegaConf
from models.model import MyNeuralNet
from pytorch_lightning.callbacks import ModelCheckpoint
from data.dataset import DogBreedsDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# TODO check pep8

# hydra configs
model_config = OmegaConf.load('ml_ops_dog_breeds/models/config.yaml')
trainer_config = OmegaConf.load('ml_ops_dog_breeds/config.yaml')

# load data
data = DogBreedsDataModule(num_workers=trainer_config.hyperparameters.num_workers)

# model class
model = MyNeuralNet(
    model_type=model_config.hyperparameters.model_type,
    out_features=model_config.hyperparameters.out_features,
    lr=trainer_config.hyperparameters.lr,
)

# monitor model checkpoints
checkpoint_callback = ModelCheckpoint(dirpath='./models', monitor='train_loss', mode='min')

# train, val datasets
train_dataloader = data.train_dataloader(batch_size=trainer_config.hyperparameters.batch_size)
# val_dataloader = data.val_dataloader(batch_size=trainer_config.hyperparameters.batch_size)

# trainer with wandb logger
trainer = Trainer(
    accelerator=trainer_config.hyperparameters.device,
    max_epochs=trainer_config.hyperparameters.n_epochs,
    callbacks=[checkpoint_callback],
    logger=WandbLogger(project='ml_ops_dog_breeds'),
)

if __name__ == '__main__':
    trainer.fit(model, train_dataloader) #, val_dataloader)
