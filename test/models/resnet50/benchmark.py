import onnx
import onnxruntime

import os
import sys
import subprocess
import time
import torch
import torch.onnx
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

#forcing thread count
os.environ["OMP_NUM_THREADS"] = "4"

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

CDBUILD = "/workdir/core-dnn/build/Debug/" #os.environ.get('CORE_DNN_BUILD_PATH')
CORE_DNN = "/workdir/core-dnn/build/Debug/bin/core-dnn" #os.path.join(CDBUILD, "bin/core-dnn")

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = "/workdir/core-dnn/build/Debug/lib" #os.path.join(CDBUILD, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import OMExecutionSession

def execute_commands(cmds):
    subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

model = torch.hub.load('pytorch/vision:v0.11.2', 'resnet50', pretrained=True) #pretrained=True)

model.eval()
print(model)

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

torch.onnx.export(model, input_batch, "resnet50.onnx", verbose=False)
onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
print("check model well formed")

execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_cudnn",
    "--cudnn-kernel-fusion=true",
    "--cudnn-dealloc-opt=false",
    "--onnx-const-hoisting=false",
    "--malloc-pool-opt=true",
    "resnet50.onnx"])

input_batch_np = input_batch.cpu().numpy()

ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("resnet50.onnx")
ort_in = {ort_session.get_inputs()[0].name: input_batch_np}

sess = OMExecutionSession("./resnet50.so")

for i in range(10):
    # using onnxruntime
    start = time.time()
    ort_out = ort_session.run(None, ort_in)
    diff = (time.time() - start)*1000
    if(i):
        ort_results.append(diff)

    # using pytorch
    model.to('cpu')
    start = time.time()
    torch_out = model(input_batch)
    diff = (time.time() - start)*1000
    if(i):
        torch_results.append(diff)

    time.sleep(0.01)
    # using occamy output
    start = time.time()
    torch.cuda.synchronize()
    occamy_out = sess.run(input_batch_np)
    torch.cuda.synchronize()
    diff = (time.time() - start)*1000
    if(i):
        occamy_results.append(diff)

    time.sleep(0.01)

    # using pytorch gpu
    if torch.cuda.is_available():
        start = time.time()
        torch.cuda.synchronize()
        torch.cuda.init()
        model_cuda = model.cuda()
        cuda_a = input_batch.cuda()
        cuda_out = model_cuda(cuda_a)
        cuda_out.to('cpu')
        del model_cuda
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        diff = (time.time() - start)*1000

        if(i):
            cuda_results.append(diff)
    time.sleep(0.02)

print(np.average(ort_results), " ",
        np.average(torch_results), " ",
        np.average(cuda_results), " ",
        np.average(occamy_results))
