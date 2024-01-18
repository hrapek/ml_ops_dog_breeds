from data.dataset import DogBreedsDataModule
from models.model import MyNeuralNet
from torchvision import transforms
import torch
import joblib
import click


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option('--checkpoint_path', default='models/epoch=29-step=3840.ckpt', help='learning rate to use for training')
@click.option('--image_path', default='data/predictions/dog.jpg', help='Image used for predictions')
def predict(checkpoint_path, image_path):
    """Predict the dog breed from an input image using a trained neural network.

    Args:
        checkpoint_path (str): Path to the model checkpoint file (default: 'models/epoch=29-step=3840.ckpt').
        image_path (str): Path to the image used for predictions (default: 'data/predictions/dog.jpg').

    Returns:
        str: Predicted dog breed label.

    """

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
