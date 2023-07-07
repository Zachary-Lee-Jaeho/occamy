import onnx
import torch
import torch.nn as nn

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Conv2d(3, 10, 3) 
    self.layer2 = nn.Conv2d(10, 10, 5)
    self.relu = nn.ReLU(inplace=False)
  def forward(self, x):
    out = self.relu(self.layer1(x))
    out = self.relu(self.layer2(out))
    return out

model = ConvNet().to("cpu")

a = torch.randn(1, 3, 64, 64)
out = model(a)

torch.onnx.export(model, a, "simpleconv.onnx", verbose=True)
onnx_model = onnx.load("simpleconv.onnx")
onnx.checker.check_model(onnx_model)
print("onnx model check")
