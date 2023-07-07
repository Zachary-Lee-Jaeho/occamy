import onnx
import onnxruntime

import torch
import torch.nn as nn

import os
import sys
import subprocess
import time
import torch
import torch.nn as nn
import numpy as np

CDBUILD = "/workdir/core-dnn/build/Debug/"
CORE_DNN = "/workdir/core-dnn/build/Debug/bin/core-dnn"

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = "/workdir/core-dnn/build/Debug/lib"
sys.path.append(RUNTIME_DIR)
from PyRuntime import OMExecutionSession

b0 = torch.randn(1, 10, 512, 512)
b1 = torch.tensor([1])

def execute_commands(cmds):
    subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Conv2d(3, 10, 3, bias=True)
    self.layer2 = nn.Conv2d(10, 10, 3, bias=True)
    self.relu = nn.ReLU(inplace=False)
  def forward(self, x):
    # out = self.relu(self.layer1(x))
    # out = self.relu(self.layer2(out))
    out = torch.add(b0, b0)
    out = torch.matmul(x, out)
    return out

model = ConvNet().to("cpu")
print(model)

a = torch.randn(1, 10, 512, 512)
a_np = a.to("cpu").numpy()

out = model(a)

torch.onnx.export(model, a, "simpleconv.onnx", verbose=True)
onnx_model = onnx.load("simpleconv.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model well formed\n\n")

execute_commands([CORE_DNN,
    "--EmitLib", "--target=nvptx_cudnn",
    "--cudnn-kernel-fusion=true",
    "--cudnn-dealloc-opt=false",
    "--onnx-const-hoisting=false",
    "--result-malloc-opt=false",
    "--malloc-pool-opt=true",
    "simpleconv.onnx"])


ort_results = []
occamy_results = []
torch_results = []
cuda_results = []

ort_session = onnxruntime.InferenceSession("simpleconv.onnx")
ort_in = {ort_session.get_inputs()[0].name: a_np}

sess = OMExecutionSession("/workdir/core-dnn/build/test/models/simpleconv/simpleconv.so")


for i in range(10):
  # # using onnxruntime
  # start = time.time()
  # ort_out = ort_session.run(None, ort_in)
  # diff = (time.time() - start)*1000
  # if(i):
  #   ort_results.append(diff)
  # else:
  #   print("ONNX RT first value : ", diff)
  #
  # # using pytorch
  # model.to('cpu')
  # torch.set_num_threads(4)
  # start = time.time()
  # torch_out = model(a)
  # diff = (time.time() - start)*1000
  # if(i):
  #   torch_results.append(diff)
  # else:
  #   print("Pytorch CPU first value : ", diff)

  # using occamy output
  time.sleep(0.01)

  start = time.time()
  occamy_out = sess.run(a_np)
  diff = (time.time() - start)*1000
  if(i):
    occamy_results.append(diff)
  else:
    print("Occamy first value : ", diff)

  time.sleep(0.01)

  # using pytorch gpu
  if torch.cuda.is_available():
    start = time.time()
    torch.cuda.synchronize()
    torch.cuda.init()
    model_cuda = model.cuda()
    b0 = b0.to("cuda:0")
    # b1 = b1.to("cuda:0")
    cuda_a = a.cuda()
    cuda_out = model_cuda(cuda_a)
    cuda_out.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    diff = (time.time() - start)*1000
    if(i):
      cuda_results.append(diff)
    else:
      print("Pytorch GPU first value : ", diff)

  time.sleep(0.02)

  del model_cuda
  model.to("cpu")
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  b0 = b0.to("cpu")
  # b1 = b1.to("cpu")

# print("\n\n================ ONNX Runtime =================")
# # print("Percentile 90 : ", np.percentile(ort_results, 90))
# # print("Percentile 95 : ", np.percentile(ort_results, 95))
# # print("Percentile 99 : ", np.percentile(ort_results, 99))
# print("Average : ", np.average(ort_results))
#
# print("================ Pytorch =================")
# # print("Percentile 90 : ", np.percentile(torch_results, 90))
# # print("Percentile 95 : ", np.percentile(torch_results, 95))
# # print("Percentile 99 : ", np.percentile(torch_results, 99))
# print("Average : ", np.average(torch_results))

print("\n\n================ Pytorch GPU =================")
# print("Percentile 90 : ", np.percentile(cuda_results, 90))
# print("Percentile 95 : ", np.percentile(cuda_results, 95))
# print("Percentile 99 : ", np.percentile(cuda_results, 99))
print("Average : ", np.average(cuda_results))

print("================ Occamy =================")
# print("Percentile 90 : ", np.percentile(occamy_results, 90))
# print("Percentile 95 : ", np.percentile(occamy_results, 95))
# print("Percentile 99 : ", np.percentile(occamy_results, 99))
print("Average ", np.average(occamy_results), "\n\n")

# ort_out_t = torch.tensor(ort_out)[0]
# occamy_out_t = torch.tensor(occamy_out)[0]
#
# t = torch.ones(ort_out_t.size())
# r = t.new_full(ort_out_t.size(), 10)
# ort_out_round = (ort_out_t * r).round()
# occamy_out_round = (occamy_out_t * r).round()
# torch_out_round = (torch_out * r).round()
# if torch.cuda.is_available():
#   cuda_out_round = (cuda_out.to("cpu") * r).round()
#
# # correctness
# if torch.all(ort_out_round.eq(occamy_out_round).reshape(-1)):
#   if torch.all(occamy_out_round.eq(torch_out_round).reshape(-1)):
#     if torch.cuda.is_available():
#       if torch.all(torch_out_round.eq(cuda_out_round).reshape(-1)):
#         print("Correctness verified.")
#       else:
#         print("Not correct output! (torch, cuda different)")
#         for i in (torch_out_round.eq(cuda_out_round).reshape(-1) == False).nonzero(as_tuple=True):
#           print(torch_out_round.reshape(-1)[i])
#           print(cuda_out_round.reshape(-1)[i])
#     else:
#       print("Correctness verified.")
#   else:
#     print("Not correct output! (occamy, torch different)")
#     cnt = 0
#     for i in (occamy_out_round.eq(torch_out_round).reshape(-1) == False).nonzero(as_tuple=True):
#       print(occamy_out_round.reshape(-1)[i])
#       print(torch_out_round.reshape(-1)[i])
#       cnt = cnt + 1
#       if cnt == 0:
#         break
# else:
#   print("Not correct output! (ort, occamy different)")
#   cnt = 0
#   for i in (ort_out_round.eq(occamy_out_round).reshape(-1) == False).nonzero(as_tuple=True):
#     print(ort_out_round.reshape(-1)[i])
#     print(occamy_out_round.reshape(-1)[i])
#     cnt = cnt + 1
#     if cnt == 10:
#         break
#
