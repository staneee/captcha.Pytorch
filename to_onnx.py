import torch
from model.BNNeck import bnneck
from model.dense import dense121
from model.dualpooling import DualResNet
from model.res18 import res18
from model.senet import senet

from config.parameters import *

if __name__ == '__main__':
    modelPath = "./weights/bnneck34_968.pth"
    savePath = "./weights/test.onnx"

    input_shape = (ImageChannel, ImageWidth, ImageWidth)

    model = bnneck()
    model.eval()
    model.train(False)
    model.load_model(modelPath)

    dummy_input = torch.randn(1,*input_shape, device='cpu')
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(model,
        dummy_input,
        savePath, 
        verbose=True, 
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes={
            "input":{0:"batch_size"},
            "output":{0:"batch_size"}
        }
    )
