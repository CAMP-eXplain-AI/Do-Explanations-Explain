from os import listdir
from os.path import isfile, join
import pickle 
import argparse

import matplotlib.pyplot as plt

def plot_heatmap(hm, dir, name):
  
  heatmap = hm.squeeze().cpu()
  if len(heatmap.shape) != 2:
  #   heatmap = heatmap.unsqueeze(2)
  # else:
    heatmap = torch.sum(heatmap, 0).squeeze()
  
  # print(heatmap.shape)
  plt.imsave(join(dir, name + '.png'), heatmap.numpy(), cmap='Reds', format='png')
  plt.imsave(join(dir, name + '.pdf'), heatmap.numpy(), cmap='Reds', format='pdf')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--whole_dir', type=bool, default=True)
  parser.add_argument('--in_dir', type=str, default='./', help='Directory to read data from.')
  parser.add_argument('--out_dir', type=str, default='./', help='Directory to save heatmaps.')

  args = parser.parse_args()

  data_dir = args.in_dir
  print(data_dir)
  data = []

  if args.whole_dir:
    hm_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
  else:
    hm_files = [data_dir]

  for file_name in hm_files:
    file_dir = join(data_dir, file_name)
    f = open(file_dir, "rb")
    # data += [pickle.load(f)]
    data = pickle.load(f)
    plot_heatmap(data, args.out_dir, file_name)
    f.close()
  



