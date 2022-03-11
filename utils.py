import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from random import shuffle
from torchvision import transforms
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def save_patch_info(closure_out):
#   p0_t0, p0_t1, p0_t2, p1_t0, p1_t1, p1_t2, patch_position, random_bg = closure_out



# Given the background and patch positions, makes the final image.
def add_patches_img(ori_img, patches):
  img = ori_img.clone()

  for i in range(len(patches)):
    x, y, pic = patches[len(patches) - i - 1]
    img[:, :, x:x+pic.shape[2], y:y+pic.shape[3]] = pic.clone()

  return img


# Takes one or two patches and, randomly places them and returns their positions.
def random_patch_placer(bg, pics):
  bgsize = bg.shape[2]
  dim = pics[0].shape[2]

  if len(pics) == 1:
    x, y = torch.randint(bgsize-dim, [2]).to(device)
    patches = [[x, y, pics[0].clone()]]

  elif len(pics) == 2:
    dim = pics[0].shape[2]
    sep = torch.randint(low=dim, high=bgsize-dim, size=[1]).item()

    x1 = torch.randint(0, sep-dim+1, [1]).item()
    x2 = torch.randint(sep, bgsize-dim+1, [1]).item()

    y1, y2 = torch.randint(0, bgsize-dim+1, [2]).to(device)
    y1, y2 = y1.item(), y2.item() 

    swap = torch.rand(1).item()
    if swap >= 0.5:
      x1, y1 = y1, x1
      x2, y2 = y2, x2

    p1, p2 = torch.clone(pics[0]), torch.clone(pics[1])
    swap_first = torch.rand(1).item()
    if swap_first >= 0.5:
      x1, x2 = x2, x1
      y1, y2 = y2, y1

    patches = [[x1, y1, p1], [x2, y2, p2]]

  elif len(pics) == 4:
    dim = pics[0].shape[2]
    sep1 = torch.randint(low=dim, high=bgsize-dim, size=[1]).item()
    sep2 = torch.randint(low=dim, high=bgsize-dim, size=[1]).item()

    x1 = torch.randint(0, sep1-dim+1, [1]).item()
    x2 = torch.randint(sep1, bgsize-dim+1, [1]).item()
    x3 = torch.randint(0, sep1-dim+1, [1]).item()
    x4 = torch.randint(sep1, bgsize-dim+1, [1]).item()

    y1 = torch.randint(0, sep2-dim+1, [1]).item()
    y2 = torch.randint(0, sep2-dim+1, [1]).item()
    y3 = torch.randint(sep2, bgsize-dim+1, [1]).item()
    y4 = torch.randint(sep2, bgsize-dim+1, [1]).item()

    XY = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    shuffle(XY)

    p1, p2, p3, p4 = torch.clone(pics[0]), torch.clone(pics[1]), torch.clone(pics[2]), torch.clone(pics[3])

    patches = [XY[0] + [p1], XY[1] + [p2], XY[2] + [p3], XY[3] + [p4]]

  return patches


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    layers = filter(lambda x: x is not None, [None, convolver, None])
    return nn.Sequential(*layers)


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


def get_resnet_preprocessor(imsize):
    preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    return preprocess


def FL(value, target, gammas):
    gamma0, gamma1 = gammas
    # global gamma0, gamma1
    p = value if target==1 else 1-value
    gamma = gamma1 if target==1 else gamma0
    return (-torch.pow(1-p, gamma)*torch.log(p)).detach().item()


## Anything below is in regard to Attribution Calculation and its concerning metrics

def image_constructor(data):
  # Returns an image of size [3, H, W] according to the data
  # data: [patch_positions, background, map_idx] 
  patch_positions, background, _ = data
  return add_patches_img(background, patch_positions)

def single_patch_image_constructor(data):
  # Returns an image of size [3, H, W] according to the data
  # data: [patch_positions, background, map_idx] 
  patch_positions, background, _ = data
  return add_patches_img(background, patch_positions[0:1])



def get_class_sensitivity(heatmaps, data):
  # heatmaps: [2, H, W], first heatmap belongs to first patch in data
  # data: [patch_positions, background, map_idx]
  patch_positions, background, map_idx = data

  patch_size = data[0][0][2].shape[2]

  first_position = data[0][0][:2]
  second_position = data[0][1][:2]

  first_hm = heatmaps[0]
  second_hm = heatmaps[1]

  # min_value = min(torch.min(first_hm), torch.min(second_hm))
  # first_hm -= min_value
  # second_hm -= min_value

  first_hm = abs(first_hm)
  second_hm = abs(second_hm)

  first_on_first_hm = first_hm[first_position[0]:first_position[0]+patch_size, first_position[1]:first_position[1]+patch_size]
  second_on_first_hm = first_hm[second_position[0]:second_position[0]+patch_size, second_position[1]:second_position[1]+patch_size]

  first_on_second_hm = second_hm[first_position[0]:first_position[0]+patch_size, first_position[1]:first_position[1]+patch_size]
  second_on_second_hm = second_hm[second_position[0]:second_position[0]+patch_size, second_position[1]:second_position[1]+patch_size]

  term_0 = torch.min(first_on_second_hm, first_on_first_hm).sum() + torch.min(second_on_first_hm, second_on_second_hm).sum()
  term_1 = first_on_first_hm.sum() + second_on_second_hm.sum()

  return term_0/term_1


def get_null_player(heatmap, data):
  # heatmap: [H, W], heatmap associated with target index
  # data: [patch_positions, background, map_idx], first entry represents target patch
  patch_positions, background, map_idx = data

  patch_size = data[0][0][2].shape[2]

  target_position = data[0][0][:2]
  null_position = data[0][1][:2]

  target_hm = heatmap[target_position[0]:target_position[0]+patch_size, target_position[1]:target_position[1]+patch_size]

  null_hm = heatmap[null_position[0]:null_position[0]+patch_size, null_position[1]:null_position[1]+patch_size]

  # Remove or Change.
  # min_value = min(torch.min(target_hm), torch.min(null_hm))
  # target_hm -= min_value
  # null_hm -= min_value

#   target_hm = abs(target_hm)
#   null_hm = abs(null_hm)

  return null_hm.sum()/target_hm.sum()


def repeated_patch_image(data):
  # Takes data and outputs an image of shape [3, H, W]
  patch = data[0][0][2]
  background = data[1]
  bgsize = background.shape[2]

  dim = patch.shape[2]
  sep1 = torch.randint(low=dim, high=bgsize-dim, size=[1]).item()
  sep2 = torch.randint(low=dim, high=bgsize-dim, size=[1]).item()

  x1 = torch.randint(0, sep1-dim+1, [1]).item()
  x2 = torch.randint(sep1, bgsize-dim+1, [1]).item()
  x3 = torch.randint(0, sep1-dim+1, [1]).item()
  x4 = torch.randint(sep1, bgsize-dim+1, [1]).item()

  y1 = torch.randint(0, sep2-dim+1, [1]).item()
  y2 = torch.randint(0, sep2-dim+1, [1]).item()
  y3 = torch.randint(sep2, bgsize-dim+1, [1]).item()
  y4 = torch.randint(sep2, bgsize-dim+1, [1]).item()

  XY = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  shuffle(XY)

  patches = [XY[0] + [patch], XY[1] + [patch], XY[2] + [patch], XY[3] + [patch]]

  return add_patches_img(background, patches)

def repeated_two_patch(data):
  # Takes data and outputs an image of shape [3, H, W]
  patch = data[0][0][2]
  background = data[1]
  bgsize = background.shape[2]

  dim = patch.shape[2]

  # x1 = torch.randint(0, 15, [1]).item()
  # y1 = torch.randint(0, 15, [1]).item()

  # x2 = torch.randint(bgsize-dim-14, bgsize-dim+1, [1]).item()
  # y2 = torch.randint(bgsize-dim-14, bgsize-dim+1, [1]).item()

  x1, y1 = 5, 5
  x2, y2 = 133, 133

  XY = [[x1, y1], [x2, y2]]
  shuffle(XY)

  patches = [XY[0] + [patch], XY[1] + [patch]]

  return add_patches_img(background, patches)


def single_patch_hm_sum(heatmap, data):
  # heatmap: [H, W], heatmap associated with target index
  # data: [patch_positions, background, map_idx], first entry represents target patch
  patch_positions, background, map_idx = data

  patch_size = data[0][0][2].shape[2]
  # position = data[0][0][:2]

  patch_position = data[0][0][:2]
  # random_position = data[0][1][:2]


  
  hm = heatmap[patch_position[0]:patch_position[0]+patch_size, patch_position[1]:patch_position[1]+patch_size]
  # random_patch = heatmap[random_position[0]:random_position[0]+patch_size, random_position[1]:random_position[1]+patch_size]

  # hm_score = torch.mean(hm)
  # hm_score = hm.mean()/random_patch.mean()
  random_mean = (heatmap.sum() - hm.sum())/(heatmap.shape[1]**2 - patch_size**2) + 0.00001
  hm_score = (hm.mean()+0.00001)/random_mean

  # return hm_score
  return hm_score, random_mean


def two_patch_corr_score(heatmap, data):
  # heatmap: [H, W], heatmap associated with target index
  # data: [patch_positions, background, map_idx], first entry represents target patch
  patch_positions, background, map_idx = data

  patch_size = data[0][0][2].shape[2]

  patch_position1 = [5, 5]
  patch_position2 = [133, 133]


  
  p1 = heatmap[patch_position1[0]:patch_position1[0]+patch_size, patch_position1[1]:patch_position1[1]+patch_size]
  p2 = heatmap[patch_position2[0]:patch_position2[0]+patch_size, patch_position2[1]:patch_position2[1]+patch_size]

  return p1.mean(), p2.mean()

#TODO: move to proper place
# corr, pval = spearmanr(x1, x2) 