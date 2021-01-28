from flask import Flask
from flask import jsonify
from flask import request
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
from predict import Dataset4Captcha, predict

import os
import base64

#
modelPath = os.getenv('model_path')
imgHeight = int(os.getenv('img_height'))
imgWidth = int(os.getenv('img_width'))

# transforms
transform = T.Compose([
    T.Resize((imgHeight, imgWidth)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# device
device = torch.device('cpu')

model = nn.Module()
modelInfo = torch.load(modelPath, map_location=device)
model.load_state_dict(modelInfo)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/recognition', methods=['POST'])
def recognition():
    data = request.get_json()
    imgStr = data['input']
    imgData = base64.decodestring(imgStr)
    return jsonify({"login": ""})
