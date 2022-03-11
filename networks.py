import torch
from skip import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

def skip_net():
  net = skip(32, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [0, 4, 4, 4, 4, 4],   
                           filter_size_down = [5, 3, 5, 5, 3, 5], filter_size_up = [5, 3, 5, 3, 5, 3], 
                           upsample_mode='bilinear', downsample_mode='avg',
                           need_sigmoid=True, pad='reflection', act_fun='LeakyReLU').type(dtype)

  removed = list(net.children())[:-1]
  net= torch.nn.Sequential(*removed)

  net.add(conv(3, 3, kernel_size=3, stride=3))
  net.add(nn.Sigmoid())
  net = net.to(device)

  return net
