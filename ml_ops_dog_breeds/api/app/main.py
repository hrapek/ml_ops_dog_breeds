from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from ml_ops_dog_breeds.models.model import MyNeuralNet
from torchvision import transforms
from ml_ops_dog_breeds.data.dataset import DogBreedsDataModule
from io import BytesIO
import joblib
import torch

app = FastAPI()

BUCKET_NAME = 'mlops-project-model-mk'
MODEL_FILE = 'models/epoch=29-step=3840.ckpt'
LABELS_FILE = 'data/processed/label_encoder.pkl'

# client = storage.Client()
# bucket = client.get_bucket(BUCKET_NAME)
# blob_model = bucket.get_blob(MODEL_FILE)
# model = MyNeuralNet.load_from_checkpoint(BytesIO(blob_model.download_as_bytes()))
# blob_data = bucket.get_blob(LABELS_FILE)
# label_encoder = joblib.load(BytesIO(blob_data.download_as_bytes()))

model = MyNeuralNet.load_from_checkpoint(MODEL_FILE)
label_encoder = joblib.load(LABELS_FILE)


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
    """Predict the dog breed from an uploaded image file.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        Dict[str, str]: Dictionary containing the predicted dog breed.

    """
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
    label = label_encoder.inverse_transform(label_num)[0]
    return {'dog_breed': label}
