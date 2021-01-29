import onnx
import torch


model = onnx.load("./weights/test.onnx")

onnx.checker.check_model(model)


onnx.helper.printable_graph(model.graph)