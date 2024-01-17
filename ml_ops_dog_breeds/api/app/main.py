from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from ml_ops_dog_breeds.models.model import MyNeuralNet
from torchvision import transforms
from ml_ops_dog_breeds.data.dataset import DogBreedsDataModule
from io import BytesIO
import joblib
import torch


app = FastAPI()


@app.get('/')
def root():
    """Health check."""
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
    }
    return response




@app.post('/predict/')
async def predict_label(file: UploadFile = File(...)):
    data = DogBreedsDataModule()

    # checkpoint location
    checkpoint_path = 'models/epoch=29-step=3840.ckpt'
    model = MyNeuralNet.load_from_checkpoint(checkpoint_path)

    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    img = await file.read()
    image_tensor = data.read_image(BytesIO(img), transformations)
    image_tensor = torch.stack([image_tensor])

    model.eval()
    pred = model(image_tensor)

    label_num = pred.argmax(dim=-1)
    label_encoder = joblib.load('data/processed/label_encoder.pkl')
    label = label_encoder.inverse_transform(label_num)[0]
    return {'dog_breed': label}
