from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
import torch.optim
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image
import pickle
import random
from tqdm import tqdm

from loss import *
from utils import *
from networks import *

from common import conv

dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Modify if testing on multiple archs.
pretrained_net = models.resnet18(pretrained=True).to(device)
pretrained_net.eval()

imsize = 227 if pretrained_net == 'alexnet_caffe' else 224

preprocess = get_resnet_preprocessor(imsize)
cnn = pretrained_net.to(device)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  # parser.add_argument('--patch_num', type=int, default=2)
  parser.add_argument('--patch_num', default='double', nargs='?', choices=['single', 'double', 'repeated', 'null'])
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--num_iter', type=int, default=1500)
  parser.add_argument('--indep_thresh', type=float, default=0.14)
  parser.add_argument('--accu_thresh', type=float, default=0.94)
  parser.add_argument('--var_thresh', type=float, default=0.0001)
  parser.add_argument('--save_as_image', type=bool, default=True, help='Save data also as an image.')
  parser.add_argument('--dir', type=str, default='./', help='Directory to save data.')
  parser.add_argument('--image_num', type=int, default=150, help='Number of images to save.')

  args = parser.parse_args()

  patch_num = args.patch_num
  LR = args.lr
  num_iter = args.num_iter
  images_saved = 0

  imsize_net = 256

  plt.style.use('seaborn')
  PLOT = False

  dir = args.dir
  data_dir = os.path.join(dir, 'Data/')
  stat_dir = os.path.join(dir,'Stat/')
  image_dir = os.path.join(dir, 'Images/')


  os.makedirs(dir, exist_ok=True)
  os.makedirs(data_dir, exist_ok=True)
  os.makedirs(stat_dir, exist_ok=True)
  if args.save_as_image:
    os.makedirs(image_dir, exist_ok=True)

  repeated = False
  if patch_num == 'single':
    closure_func = closure_single_patch
    patch_num = 1
  elif patch_num == 'double':
    closure_func = closure
    patch_num = 2
  elif patch_num == 'null':
    closure_func = closure_null
    patch_num = 2
  elif patch_num == 'repeated':
    repeated = True
    closure_func = closure_repeated_patch
    patch_num = 2

  while images_saved < args.image_num:

    prior_nets = [skip_net() for i in range(patch_num)]
    net_inputs = [get_noise(32, 'noise', imsize_net).type(dtype).detach() for i in range(patch_num)]
    optimizer = torch.optim.RMSprop([item for i in range(patch_num) for item in prior_nets[i].parameters()], lr=LR)

    counter = 0
    last_iters = []

    map_idxs = random.sample(range(0, 500), 1) + random.sample(range(500, 1000), 1)
    print("labels:", map_idxs)

    key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1 = 0.5, 0.5, 0.5, 0.5
    alpha, gamma0, gamma1 = 0.1, 0.8, 0.8

    for j in tqdm(range(num_iter)):
      optimizer.zero_grad()

      if args.patch_num == 'single':     
        keys = key_p0_t0, key_p0_t1
      else:
        keys = key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1

      p0, p1, pic, ALL, patch_positions, random_bg, keys = closure_func(cnn, preprocess, prior_nets, net_inputs, patch_num, imsize, map_idxs, j, keys)

      if args.patch_num == 'single':     
        key_p0_t0, key_p0_t1 = keys
      else:
        key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1 = keys

      last_iters += p0
      if all(x < args.indep_thresh for x in p1) and j > 50 and all(args.accu_thresh < x for x in p0):
        images_saved += 1

        f =  open( stat_dir + "stat." + str(images_saved) + ".txt","w+")
        f.write("indexes: " + str(map_idxs) + "\n")
        f.write("p0: " + str(p0) + "\n")  
        f.write("p1: " + str(p1) + "\n")
        f.write("scores: " + " ".join(str(a) for a in ALL))
        f.close()


        data_file = os.path.join(data_dir, 'data.' + str(images_saved))
        f = open(data_file, 'wb')
        pickle.dump([patch_positions, random_bg, map_idxs], f)
        f.close()

        # saving figure
        if args.save_as_image:
          image_file = os.path.join(image_dir, "image."+str(images_saved)+ "_" +"_".join(str(v) for v in map_idxs)+".pdf")
          save_image(pic, image_file)
        print("SAVED!")
        break;

      optimizer.step()

    print("Done. Images saved: "+ str(images_saved))    
