import torch
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import numpy as np
import joblib


class Handler(BaseHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._context = None
        self.initialized = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self, context):
        '''load the model and the label encoder'''

        self._context = context
        properties = context.system_properties
        model_dir = properties.get('model_dir')
        self.model = torch.jit.load(model_dir + '/model.pt')
        self.model.eval()
        self.label_encoder = joblib.load(model_dir + '/label_encoder.pkl')
        self.initialized = True

    def preprocess(self, data):
        '''convert the image file to a tensor and apply transformations'''

        data = data[0].get('data') or data[0].get('body')
        image = Image.open(io.BytesIO(data))
        image = image.resize((224, 224))
        image = np.array(image).transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        image = image / 255.0
        return image

    def inference(self, data):
        '''predict the class of the image'''

        output = self.model(data.unsqueeze(0))
        output = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(output, 5)
        top_labels = self.label_encoder.inverse_transform(top_indices[0].numpy())
        top_probs_list = [round(prob.item(), 2) for prob in top_probs[0]]
        result = [{'top_breeds': [{'breed': label, 'probability': prob} for label, prob in zip(top_labels, top_probs_list)]}]
        return result

    def postprocess(self, data):
        '''process the predictions and return the result'''
        return data
