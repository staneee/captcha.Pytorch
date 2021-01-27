import csv
import os

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms as T

from config.parameters import *
from lib.dataset import *
from model.BNNeck import bnneck
from model.dense import dense121
from model.dualpooling import DualResNet
from model.model import *
from model.res18 import res18
from model.senet import senet
from predict import Dataset4Captcha, predict, predict_all
from train import *

if __name__ == '__main__':
    modelPath1 = "./weights/bnneck25_927.pth"
    modelPath2 = "./weights/bnneck34_968.pth"
    dataInputPath = r""


    # device = torch.device('cpu')
    
    model = bnneck()
    model.eval()
    
    # model.load_model(modelPath1)
    model.load_model(modelPath2)
    # if t.cuda.is_available():
    #     model = model.cuda()
    inputDataset = Dataset4Captcha(dataInputPath, train=False)
    inputDataLoader = DataLoader(inputDataset, batch_size=1,
                                    shuffle=False, num_workers=1)     
    print("start run")
    predict(model,inputDataLoader,"result/tmp.csv")   
