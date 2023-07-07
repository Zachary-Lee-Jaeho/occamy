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

def execute_commands(cmds):
    subprocess.run(cmds, stdout=subprocess.PIPE).check_returncode()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 10, 3, padding=1)
        self.layer2 = nn.Conv2d(10, 10, 5, padding=2)
        self.layer3 = nn.Conv2d(10, 10, 5, padding=2)
        self.layer4 = nn.Conv2d(10, 10, 5, padding=2)
        self.layer5 = nn.Conv2d(10, 10, 5, padding=2)
        self.layer6 = nn.Conv2d(10, 10, 5, padding=2)
        self.layer7 = nn.Conv2d(10, 10, 5, padding=2)
        self.layer8 = nn.Conv2d(10, 10, 5, padding=2)
        self.relu = nn.ReLU(inplace=False)
        self.FC = nn.Linear(64, 1000)
    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.relu(self.layer3(out))
        out = self.relu(self.layer4(out)) + out
        out = self.relu(self.layer5(out))
        out = self.relu(self.layer6(out))
        out = self.relu(self.layer7(out)) + out
        out = self.FC(out)
        return out

model = ConvNet().to("cpu")
print(model)

a = torch.randn(1, 3, 64, 64)
a_np = a.to("cpu").numpy()

out = model(a)

torch.onnx.export(model, a, "simpleconv.onnx", verbose=True)
onnx_model = onnx.load("simpleconv.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model well formed\n\n")

execute_commands([CORE_DNN, "--EmitLib", "--target=nvptx_cudnn","--cudnn-kernel-fusion=false", "simpleconv.onnx"])
execute_commands([CORE_DNN, "--EmitLib", "--target=nvptx_cudnn", "simpleconv.onnx", "-o", "simpleconv_fused"])

sess = OMExecutionSession("/workdir/core-dnn/build/test/models/simpleconv/simpleconv.so")
sess_fused = OMExecutionSession("/workdir/core-dnn/build/test/models/simpleconv/simpleconv_fused.so")

num_iteration = 10000
occamy_avg = 0
occamy_fused_avg = 0

for i in range(num_iteration):
    if (i%100 == 0):
        print("Iteration :", i)

    # using fusded occamy output
    start = time.time()
    occamy_out = sess.run(a_np)
    diff = (time.time() - start)*1000
    if(i<3):
        print("Occamy time : ", diff)
    else:
        occamy_avg = occamy_avg + (diff/num_iteration)

    # using fusded occamy output
    start = time.time()
    occamy_out = sess_fused.run(a_np)
    diff = (time.time() - start)*1000
    if(i<3):
        print("Occamy Fused time : ", diff)
    else:
        occamy_fused_avg = occamy_fused_avg + (diff/num_iteration)


print("================ Occamy =================")
print("Average ", occamy_avg, "\n\n")

print("============= Occamy Fused ==============")
print("Average ", occamy_fused_avg, "\n\n")

