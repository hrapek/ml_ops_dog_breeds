FROM python:3.11-slim
WORKDIR /code
COPY ./requirements_api.txt /code/requirements_api.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements_api.txt
COPY ml_ops_dog_breeds/ ml_ops_dog_breeds/
COPY models/ models/
COPY data/processed/label_encoder.pkl data/processed/label_encoder.pkl

CMD ["uvicorn", "ml_ops_dog_breeds.api.app.main:app", "--host", "0.0.0.0", "--port", "80"]
