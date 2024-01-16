from data.dataset import DogBreedsDataModule
from models.model import MyNeuralNet
from torchvision import transforms
import torch
import joblib
import click

# TODO: figure out how to load models, lighting saves model checkpoint but with weird names (how to make it easier?)


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option('--checkpoint_path', default='models/epoch=0-step=128-v1.ckpt', help='learning rate to use for training')
@click.option('--image_path', default='data/predictions/dog.jpg', help='Image used for predictions')
def predict(checkpoint_path, image_path):
    # data
    data = DogBreedsDataModule()

    # checkpoint location
    model = MyNeuralNet.load_from_checkpoint(checkpoint_path)

    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image_tensor = data.read_image(image_path, transformations)
    image_tensor = torch.stack([image_tensor])

    model.eval()
    pred = model(image_tensor)

    label_num = pred.argmax(dim=-1)
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    label = label_encoder.inverse_transform(label_num)
    return label


if __name__ == '__main__':
    predict()
