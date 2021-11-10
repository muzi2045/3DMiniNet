import torch
from torch import nn
import numpy as np

class Demo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=3, padding=1)

    def forward(self, x):
        n,c,h,w = x.shape
        unfold_x = self.unfold(x).view(n,-1,h,w)
        return unfold_x

if __name__ == "__main__":
    input_tensor = torch.zeros((1,16,100,100))
    demo = Demo()
    out = demo(input_tensor)
    torch.onnx.export(demo, input_tensor, "debug.onnx", verbose=True,
                        input_names=['data'],
                        opset_version=11
                        # do_constant_folding=True,
                        # dynamic_axes={'data':{0:'batch', 2:'width', 3:'height'},
                        # }
                        )