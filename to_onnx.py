import torch
from model.BNNeck import bnneck
from model.dense import dense121
from model.dualpooling import DualResNet
from model.res18 import res18
from model.senet import senet

from config.parameters import *

import torch.utils.model_zoo as model_zoo

if __name__ == '__main__':
    modelPath = "./weights/bnneck34_968.pth"
    savePath = "./weights/test.onnx"

    input_shape = (ImageChannel,  ImageHeight, ImageWidth)

    model = bnneck()
    model.load_state_dict(torch.load(modelPath, map_location='cpu'))
    model.train(False)

    dummy_input = torch.randn(
        1, *input_shape, device='cpu', requires_grad=True)
    input_names = ["input"]
    output_names = ["a", "b", "c", "d"]

    torch.onnx.export(model,
                      dummy_input,
                      savePath,
                      verbose=True,
                      export_params=True,
                      input_names=input_names,
                      output_names=output_names,
                      keep_initializers_as_inputs=True
                      )
