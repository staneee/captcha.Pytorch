import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

from PIL import Image
from torchvision import transforms

import numpy as np

ImageHeight = 26
ImageWidth = 87

transform = transforms.Compose([
    transforms.Resize((ImageHeight, ImageWidth)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

imgPath = r"C:\Users\yihan\Desktop\captcha_server\11\HHRJ_0a746168-94a9-482f-90e5-78ccff944311.jpg"

img = Image.open(imgPath)

data = transform(img).view(1, 3,  26, 87).numpy().astype(np.float32)


onnxModel = onnx.load("./weights/test.onnx")

# print(onnxModel.graph.input)
# print(data.shape)
# exit()
# W = {onnxModel.graph.input[0].name: data}
model = onnx_caffe2_backend.prepare(onnxModel)


def max_indices(arr, k):
    '''
    Returns the indices of the k first largest elements of arr
    (in descending order in values)
    '''
    assert k <= arr.size, 'k should be smaller or equal to the array size'
    arr_ = arr.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = np.max(arr_)
        if np.isinf(max_element):
            break
        else:
            idx = np.where(arr_ == max_element)
        max_idxs.append(idx)
        arr_[idx] = -np.inf
    return max_idxs


a, b, c, d = model.run(data)
print(max_indices(a, 1))
print(max_indices(b, 1))
print(max_indices(c, 1))
print(max_indices(d, 1))
# print(a[0])
# print(b)
# print(c)
# print(d)

# partition_arg_topK(a[0], 1, 1)
