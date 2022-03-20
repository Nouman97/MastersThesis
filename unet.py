import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def forward(self, x):
    return self.conv(x)
  
class UNET(nn.Module):
  def __init__(self, in_channels = 1, num_classes = 5, features = [64, 128, 256, 512]):
    super(UNET, self).__init__()

    # Down

    self.d1 = DoubleConv(in_channels, features[0])
    self.d2 = DoubleConv(features[0], features[1])
    self.d3 = DoubleConv(features[1], features[2])
    self.d4 = DoubleConv(features[2], features[3])

    # Bottleneck

    self.d5 = DoubleConv(features[3], features[3] * 2)

    # Up

    self.u1 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, 2)
    self.du1 = DoubleConv(features[3] * 2, features[3])

    self.u2 = nn.ConvTranspose2d(features[2] * 2, features[2], 2, 2)
    self.du2 = DoubleConv(features[2] * 2, features[2])

    self.u3 = nn.ConvTranspose2d(features[1] * 2, features[1], 2, 2)
    self.du3 = DoubleConv(features[1] * 2, features[1])

    self.u4 = nn.ConvTranspose2d(features[0] * 2, features[0], 2, 2)
    self.du4 = DoubleConv(features[0] * 2, features[0])

    # Output Layer

    self.out = nn.Conv2d(features[0], num_classes, kernel_size = 1)

    # Other Layers

    self.pool = nn.MaxPool2d(2, 2)

  def forward(self, x):

    d1 = self.d1(x) # 256 x 256 x 1 => 256 x 256 x 64
    d1_ = self.pool(d1) # 256 x 256 x 64 => 128 x 128 x 64
    d2 = self.d2(d1_) # 128 x 128 x 64 => 128 x 128 x 128
    d2_ = self.pool(d2) # 128 x 128 x 128 => 64 x 64 x 128
    d3 = self.d3(d2_) # 64 x 64 x 128 => 64 x 64 x 256
    d3_ = self.pool(d3) # 64 x 64 x 256 => 32 x 32 x 256
    d4 = self.d4(d3_) # 32 x 32 x 256 => 32 x 32 x 512
    d4_ = self.pool(d4) # 32 x 32 x 512 => 16 x 16 x 512

    d5 = self.d5(d4_) # 16 x 16 x 512 => 16 x 16 x 1024

    u1 = self.u1(d5) # 16 x 16 x 1024 => 32 x 32 x 512
    t1 = torch.cat((d4, u1), 1) # 32 x 32 x 512 => 32 x 32 x 1024
    t1 = self.du1(t1) # 32 x 32 x 1024 => 32 x 32 x 512

    u2 = self.u2(t1) # 32 x 32 x 512 => 64 x 64 x 256
    t2 = torch.cat((d3, u2), 1) # 64 x 64 x 256 => 64 x 64 x 512
    t2 = self.du2(t2) # 64 x 64 x 512 => 64 x 64 x 256

    u3 = self.u3(t2) # 64 x 64 x 256 => 128 x 128 x 128
    t3 = torch.cat((d2, u3), 1) # 128 x 128 x 128 => 128 x 128 x 256
    t3 = self.du3(t3) # 128 x 128 x 256 => 128 x 128 x 128

    u4 = self.u4(t3) # 128 x 128 x 128 => 256 x 256 x 64
    t4 = torch.cat((d1, u4), 1) # 256 x 256 x 64 => 256 x 256 x 128
    t4 = self.du4(t4) # 256 x 256 x 64 => 256 x 256 x 64

    out = self.out(t4)

    return out
