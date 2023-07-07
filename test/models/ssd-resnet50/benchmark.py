import os
import torch

#forcing thread count
os.environ["OMP_NUM_THREADS"] = "4"

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

import onnx
import onnxruntime

import sys
import subprocess
import time
import torch.onnx
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

CDBUILD = os.environ.get('CORE_DNN_BUILD_PATH')
CORE_DNN = os.path.join(CDBUILD, "bin/core-dnn")

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(CDBUILD, "lib")
sys.path.append(RUNTIME_DIR)
from PyRuntime import OMExecutionSession

def execute_commands(cmds):
  subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

precision = 'fp32'
#ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision, map_location=torch.device('cpu'))
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

uris = [
  'http://images.cocodataset.org/val2017/000000397133.jpg',
  'http://images.cocodataset.org/val2017/000000037777.jpg',
  'http://images.cocodataset.org/val2017/000000252219.jpg'
]

inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs, precision == 'fp16')

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    ssd_model.to('cuda')

with torch.no_grad():
    detections_batch = ssd_model(tensor)

ssd_model.eval()

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

classes_to_labels = utils.get_coco_object_dictionary()

torch.onnx.export(ssd_model, tensor, "ssd-resnet50.onnx", verbose=True)
onnx_model = onnx.load("ssd-resnet50.onnx")
onnx.checker.check_model(onnx_model)
print("check model well formed")

execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_cudnn",
    "--cudnn-kernel-fusion=true",
    "--cudnn-dealloc-opt=false",
    "--onnx-const-hoisting=false",
    "--result-malloc-opt=false",
    "--malloc-pool-opt=true",
    "ssd-resnet50.onnx"])

input_batch_np = tensor.cpu().numpy()

ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("ssd-resnet50.onnx")
ort_in = {ort_session.get_inputs()[0].name: input_batch_np}

sess = OMExecutionSession("./ssd-resnet50.so")

for i in range(100):
    # using onnxruntime
    start = time.time()
    ort_out = ort_session.run(None, ort_in)
    diff = (time.time() - start)*1000
    if(i):
        ort_results.append(diff)

    # using pytorch
    ssd_model.to('cpu')
    cpu_a = tensor.to('cpu')
    torch.set_num_threads(4)
    start = time.time()
    with torch.no_grad():
        torch_out = ssd_model(cpu_a)
    diff = (time.time() - start)*1000
    if(i):
        torch_results.append(diff)

    # using occamy output
    start = time.time()
    occamy_out = sess.run(input_batch_np)
    diff = (time.time() - start)*1000
    if(i):
        occamy_results.append(diff)

    # using pytorch gpu
    if torch.cuda.is_available():
        start = time.time()
        torch.cuda.synchronize()
        torch.cuda.init()
        ssd_model_cuda = ssd_model.cuda()
        cuda_a = tensor.cuda()
        cuda_out = ssd_model_cuda(cuda_a)
        cuda_out = tuple(te.cpu() for te in cuda_out)
        del ssd_model_cuda
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        diff = (time.time() - start)*1000

        if(i):
            cuda_results.append(diff)

print(np.average(ort_results), " ",
        np.average(torch_results), " ",
        np.average(cuda_results), " ",
        np.average(occamy_results))

