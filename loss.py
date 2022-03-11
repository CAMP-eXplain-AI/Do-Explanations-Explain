import torch
from utils import *

dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha, gamma0, gamma1 = 0.1, 0.8, 0.8
# batch_size while optimization is 1. changing this may increase efficiency.

def closure(cnn, preprocess, prior_nets, net_inputs, patch_num, imsize, map_idxs, counter, keys):

#   global key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1
  global alpha, gamma0, gamma1
  key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1 = keys

  patches = [prior_nets[i](net_inputs[i]) for i in range(patch_num)]  

  random_bg = torch.clamp(torch.zeros([1, 3, imsize, imsize]).normal_(mean=0.5, std=0.1).type(dtype), 0, 1).to(device)
  random_bg_out = cnn(preprocess(random_bg)).to(device)
  patch_positions = random_patch_placer(random_bg, patches)

  single_patch_pics = []
  single_patch_out = []
  for i in range(patch_num):
    pic = add_patches_img(random_bg, [patch_positions[i]])
    single_patch_pics.append(pic)
    single_patch_out.append(cnn(preprocess(pic)))
  
  combined_patch_pic0 = add_patches_img(single_patch_pics[1].clone().detach(), [patch_positions[0]])
  combined_patch_out0 = cnn(preprocess(combined_patch_pic0))

  combined_patch_pic1 = add_patches_img(single_patch_pics[0].clone().detach(), [patch_positions[1]])
  combined_patch_out1 = cnn(preprocess(combined_patch_pic1))

  p0_t0 = single_patch_out[0][0, map_idxs[0]]
  p0_t1 = (combined_patch_out0[0, map_idxs[1]] - single_patch_out[1][0, map_idxs[1]].clone().detach())**2

  p1_t0 = single_patch_out[1][0, map_idxs[1]]
  p1_t1 = (combined_patch_out1[0, map_idxs[0]] - single_patch_out[0][0, map_idxs[0]].clone().detach())**2


  m = nn.Softmax(dim=1)
  normalized_single_patch_out = [m(single_patch_out[i]).clone().detach() for i in range(patch_num)]
  
  normalized_combined_patch_out0 = m(combined_patch_out0).clone().detach()
  normalized_combined_patch_out1 = m(combined_patch_out1).clone().detach()

  # TODO: We may want to adjust p{i}_t1 terms. Dropin 20 lower bound would make the explanation more persuasive and perhaps the optimization easier  
  with torch.no_grad():
      normalized_p0_t0 = normalized_single_patch_out[0][0, map_idxs[0]]
      normalized_p0_t1 = torch.tanh(abs(combined_patch_out0[0, map_idxs[1]] - single_patch_out[1][0, map_idxs[1]])
        /min(abs(combined_patch_out0[0, map_idxs[1]].item()), abs(single_patch_out[1][0, map_idxs[1]]))).clone().detach()
        
      normalized_p1_t0 = normalized_single_patch_out[1][0, map_idxs[1]]
      normalized_p1_t1 = torch.tanh(abs(combined_patch_out1[0, map_idxs[0]] - single_patch_out[0][0, map_idxs[0]])
        /min(abs(combined_patch_out1[0, map_idxs[0]].item()), abs(single_patch_out[0][0, map_idxs[0]]))).clone().detach()

      if counter > 300:
        key_p0_t0 = normalized_p0_t0 * alpha + (1 - alpha) * key_p0_t0
        key_p1_t0 = normalized_p1_t0 * alpha + (1 - alpha) * key_p1_t0
        key_p0_t1 = normalized_p0_t1 * alpha + (1 - alpha) * key_p0_t1
        key_p1_t1 = normalized_p1_t1 * alpha + (1 - alpha) * key_p1_t1


  if counter <= 300:
    loss_patch0 = -5 * p0_t0
    loss_patch1 = -5 * p1_t0
  else:
    gammas = [gamma0, gamma1]
    loss_patch0 = -10 * FL(key_p0_t0, 1, gammas) * p0_t0 + 5 * FL(key_p0_t1, 0, gammas) * p0_t1
    loss_patch1 = -10 * FL(key_p1_t0, 1, gammas) * p1_t0 + 5 * FL(key_p1_t1, 0, gammas) * p1_t1 

  # Back Prop
  loss_patch0.backward()
  loss_patch1.backward()
  
  P0 = [normalized_p0_t0.item(), normalized_p1_t0.item()]
  P1 = [normalized_p0_t1.item(), normalized_p1_t1.item()]
  SCORES = [single_patch_out[i][0, map_idxs[i]].item() for i in range(patch_num)]  

  keys = key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1
  return P0, P1, combined_patch_pic0, SCORES, patch_positions, random_bg, keys


def closure_null(cnn, preprocess, prior_nets, net_inputs, patch_num, imsize, map_idxs, counter, keys):
#   global key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1
  global alpha, gamma0, gamma1
  key_p0_t0, key_p0_t1, key_p0_t2, key_p1_t0 = keys

  patches = [prior_nets[i](net_inputs[i]) for i in range(patch_num)]  

  random_bg = torch.clamp(torch.zeros([1, 3, imsize, imsize]).normal_(mean=0.5, std=0.1).type(dtype), 0, 1).to(device)
  random_bg_out = cnn(preprocess(random_bg)).to(device)
  patch_positions = random_patch_placer(random_bg, patches)

  single_patch_pics = []
  single_patch_out = []
  for i in range(patch_num):
    pic = add_patches_img(random_bg, [patch_positions[i]])
    single_patch_pics.append(pic)
    single_patch_out.append(cnn(preprocess(pic)))
  
  combined_patch_pic0 = add_patches_img(single_patch_pics[1].clone().detach(), [patch_positions[0]])
  combined_patch_out0 = cnn(preprocess(combined_patch_pic0))

  combined_patch_pic1 = add_patches_img(single_patch_pics[0].clone().detach(), [patch_positions[1]])
  combined_patch_out1 = cnn(preprocess(combined_patch_pic1))

  p0_t0 = single_patch_out[0][0, map_idxs[0]]
  p0_t1 = (combined_patch_out0[0, map_idxs[1]] - single_patch_out[1][0, map_idxs[1]].clone().detach())**2
  p0_t2 = (single_patch_out[0][0, map_idxs[1]] - random_bg_out[0, map_idxs[1]].clone().detach())**2

  p1_t0 = single_patch_out[1][0, map_idxs[1]]
  p1_t1 = (combined_patch_out1[0, map_idxs[0]] - single_patch_out[0][0, map_idxs[0]].clone().detach())**2


  m = nn.Softmax(dim=1)
  normalized_single_patch_out = [m(single_patch_out[i]) for i in range(patch_num)]
  
  normalized_combined_patch_out0 = m(combined_patch_out0).clone().detach()
  normalized_combined_patch_out1 = m(combined_patch_out1).clone().detach()

  # TODO: We may want to adjust p{i}_t1 terms. Dropin 20 lower bound would make the explanation more persuasive and perhaps the optimization easier  
  normalized_p1_t0 = normalized_single_patch_out[1][0, map_idxs[1]]
  with torch.no_grad():
      normalized_p0_t0 = normalized_single_patch_out[0][0, map_idxs[0]]
      normalized_p0_t1 = torch.tanh(abs(combined_patch_out0[0, map_idxs[1]] - single_patch_out[1][0, map_idxs[1]])
        /min(abs(combined_patch_out0[0, map_idxs[1]].item()), abs(single_patch_out[1][0, map_idxs[1]]))).clone().detach()
      normalized_p0_t2 = torch.tanh(abs(single_patch_out[0][0, map_idxs[1]] - random_bg_out[0, map_idxs[1]])
        /min(abs(random_bg_out[0, map_idxs[1]].item()), abs(single_patch_out[0][0, map_idxs[1]]))).clone().detach()

      if counter > 300:
        key_p0_t0 = normalized_p0_t0 * alpha + (1 - alpha) * key_p0_t0
        key_p0_t1 = normalized_p0_t1 * alpha + (1 - alpha) * key_p0_t1
        key_p0_t2 = normalized_p0_t2 * alpha + (1 - alpha) * key_p0_t2
        key_p1_t0 = normalized_p1_t0 * alpha + (1 - alpha) * key_p1_t0


  if counter <= 300:
    loss_patch0 = -5 * p0_t0
    loss_patch1 = 5 * (normalized_p1_t0 - 0.95)**2
  else:
    gammas = [gamma0, gamma1]
    loss_patch0 = -10 * FL(key_p0_t0, 1, gammas) * p0_t0 + 5 * FL(key_p0_t1, 0, gammas) * p0_t1 + 5 * FL(key_p0_t2, 0, gammas) * p0_t2
    loss_patch1 = 5 * (normalized_p1_t0 - 0.95)**2

  # Back Prop
  loss_patch0.backward()
  loss_patch1.backward()
  
  P0 = [normalized_p0_t0.item(), normalized_p1_t0.item()]
  P1 = [normalized_p0_t1.item(), normalized_p0_t2.item()]
  SCORES = [single_patch_out[i][0, map_idxs[i]].item() for i in range(patch_num)]  

  keys = key_p0_t0, key_p0_t1, key_p0_t2, key_p1_t0
  return P0, P1, combined_patch_pic0, SCORES, patch_positions, random_bg, keys
    


def closure_single_patch(cnn, preprocess, prior_nets, net_inputs, patch_num, imsize, map_idxs, counter, keys):

#   global key_p0_t0, key_p0_t1
  global alpha, gamma0, gamma1
  key_p0_t0, key_p0_t1 = keys

  patches = [prior_nets[i](net_inputs[i]) for i in range(patch_num)]  

  random_bg = torch.clamp(torch.zeros([1, 3, imsize, imsize]).normal_(mean=0.5, std=0.1).type(dtype), 0, 1).to(device)
  random_bg_out = cnn(preprocess(random_bg)).to(device)
  patch_positions = random_patch_placer(random_bg, patches)

  single_patch_pics = []
  single_patch_out = []
  for i in range(patch_num):
    pic = add_patches_img(random_bg, [patch_positions[i]])
    single_patch_pics.append(pic)
    single_patch_out.append(cnn(preprocess(pic)))
  
  p0_t0 = single_patch_out[0][0, map_idxs[0]]
  p0_t1 = (single_patch_out[0][0, map_idxs[1]] - random_bg_out[0, map_idxs[1]].clone().detach())**2


  m = nn.Softmax(dim=1)
  normalized_single_patch_out = [m(single_patch_out[i]).clone().detach() for i in range(patch_num)]

  # TODO: We may want to adjust p{i}_t1 terms. Dropin 20 lower bound would make the explanation more persuasive and perhaps the optimization easier  
  with torch.no_grad():
      normalized_p0_t0 = normalized_single_patch_out[0][0, map_idxs[0]]
      normalized_p0_t1 = torch.tanh(abs(single_patch_out[0][0, map_idxs[1]] - random_bg_out[0, map_idxs[1]])
        /min(abs(random_bg_out[0, map_idxs[1]].item()), abs(single_patch_out[0][0, map_idxs[1]]))).clone().detach()
        
      if counter > 300:
        key_p0_t0 = normalized_p0_t0 * alpha + (1 - alpha) * key_p0_t0
        key_p0_t1 = normalized_p0_t1 * alpha + (1 - alpha) * key_p0_t1


  if counter <= 300:
    loss_patch0 = -5 * p0_t0
  else:
    gammas = [gamma0, gamma1]
    loss_patch0 = -10 * FL(key_p0_t0, 1, gammas) * p0_t0 + 5 * FL(key_p0_t1, 0, gammas) * p0_t1

  # Back Prop
  loss_patch0.backward()
  
  P0 = [normalized_p0_t0.item()]
  P1 = [normalized_p0_t1.item()]
  SCORES = [single_patch_out[i][0, map_idxs[i]].item() for i in range(patch_num)]  


  keys = key_p0_t0, key_p0_t1
  return P0, P1, single_patch_pics[0], SCORES, patch_positions, random_bg, keys
  
  
def closure_repeated_patch(cnn, preprocess, prior_nets, net_inputs, patch_num, imsize, map_idxs, counter, keys):
#   global key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1
  global alpha, gamma0, gamma1
  key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1 = keys

  patches = [prior_nets[i](net_inputs[i]) for i in range(patch_num)]  

  map_idxs = [map_idxs[0], map_idxs[0]]

  random_bg = torch.clamp(torch.zeros([1, 3, imsize, imsize]).normal_(mean=0.5, std=0.1).type(dtype), 0, 1).to(device)
  random_bg_out = cnn(preprocess(random_bg)).to(device)
  patch_positions = random_patch_placer(random_bg, patches)
  patch_positions[0][:2] = [5, 5]
  patch_positions[1][:2] = [133, 133]

  single_patch_pics = []
  single_patch_out = []
  for i in range(patch_num):
    pic = add_patches_img(random_bg, [patch_positions[i]])
    single_patch_pics.append(pic)
    single_patch_out.append(cnn(preprocess(pic)))
  
  combined_patch_pic0 = add_patches_img(single_patch_pics[1].clone().detach(), [patch_positions[0]])
  combined_patch_out0 = cnn(preprocess(combined_patch_pic0))

  combined_patch_pic1 = add_patches_img(single_patch_pics[0].clone().detach(), [patch_positions[1]])
  combined_patch_out1 = cnn(preprocess(combined_patch_pic1))

  p0_t0 = single_patch_out[0][0, map_idxs[0]]
  p0_t1 = (combined_patch_out0[0, map_idxs[1]] - single_patch_out[1][0, map_idxs[1]].clone().detach())**2

  p1_t0 = single_patch_out[1][0, map_idxs[1]]
  p1_t1 = (combined_patch_out1[0, map_idxs[0]] - single_patch_out[0][0, map_idxs[0]].clone().detach())**2


  m = nn.Softmax(dim=1)
  normalized_single_patch_out = [m(single_patch_out[i]) for i in range(patch_num)]
  
  normalized_combined_patch_out0 = m(combined_patch_out0).clone().detach()
  normalized_combined_patch_out1 = m(combined_patch_out1).clone().detach()

  # TODO: We may want to adjust p{i}_t1 terms. Dropin 20 lower bound would make the explanation more persuasive and perhaps the optimization easier  
  with torch.no_grad():
    normalized_p0_t0 = normalized_single_patch_out[0][0, map_idxs[0]]
    normalized_p0_t1 = torch.tanh(abs(combined_patch_out0[0, map_idxs[1]] - single_patch_out[1][0, map_idxs[1]])
      /min(abs(combined_patch_out0[0, map_idxs[1]].item()), abs(single_patch_out[1][0, map_idxs[1]]))).clone().detach()
      
    normalized_p1_t0 = normalized_single_patch_out[1][0, map_idxs[1]]
    normalized_p1_t1 = torch.tanh(abs(combined_patch_out1[0, map_idxs[0]] - single_patch_out[0][0, map_idxs[0]])
      /min(abs(combined_patch_out1[0, map_idxs[0]].item()), abs(single_patch_out[0][0, map_idxs[0]]))).clone().detach()

    if counter > 200:
      key_p0_t0 = normalized_p0_t0 * alpha + (1 - alpha) * key_p0_t0
      key_p1_t0 = normalized_p1_t0 * alpha + (1 - alpha) * key_p1_t0
      key_p0_t1 = normalized_p0_t1 * alpha + (1 - alpha) * key_p0_t1
      key_p1_t1 = normalized_p1_t1 * alpha + (1 - alpha) * key_p1_t1


  if counter <= 200:
    loss_patch0 = -5 * p0_t0
    loss_patch1 = -5 * p1_t0
  else:
    gammas = [gamma0, gamma1]    
    loss_patch0 = -10 * FL(key_p0_t0, 1, gammas) * p0_t0 + 5 * FL(key_p0_t1, 0, gammas) * p0_t1
    loss_patch1 = -10 * FL(key_p1_t0, 1, gammas) * p1_t0 + 5 * FL(key_p1_t1, 0, gammas) * p1_t1 

  # Back Prop
  loss_patch0.backward()
  loss_patch1.backward()
  
  P0 = [normalized_p0_t0.item(), normalized_p1_t0.item()]
  P1 = [normalized_p0_t1.item(), normalized_p1_t1.item()]
  SCORES = [single_patch_out[i][0, map_idxs[i]].item() for i in range(patch_num)]  
  keys = key_p0_t0, key_p0_t1, key_p1_t0, key_p1_t1

  return P0, P1, combined_patch_pic0, SCORES, patch_positions, random_bg, keys