import random
import os
import numpy as np
import socket
import torch
from scipy import misc

def download_data(path):
    full_path = '%s/suncg/single_object/chair/' % path
    if not os.path.isdir(full_path):
        print('Downloading SUNCG chair dataset to %s' % path)
        import pdb; pdb.set_trace()
        url = 'http://www.cs.nyu.edu/~denton/datasets/suncg_chairs.tar.gz'
        os.system('wget -O %s/suncg_data.tar.gz %s' % (path, url))
        os.system('tar -xzvf %s/suncg_data.tar.gz -C %s' % (path, path))
    return full_path

class SUNCG(object):

  def __init__(self, train, data_root, seq_len = 20, image_size=64):
    self.data_root = download_data(data_root)
    self.seq_len = seq_len
    self.image_size = image_size 

    self.elevations = ['elevation_1.25', 'elevation_-1.00', 'elevation_0.50', 'elevation_2.00', 'elevation_-0.25']
    self.dirs = os.listdir(self.data_root)
    if train:
      self.start_idx = 0
      self.stop_idx = int(len(self.dirs)*0.85)
      print('Loaded train data(%d - %d objects)' % (self.start_idx, self.stop_idx))
    else:
      self.start_idx = int(len(self.dirs)*0.85)+1 
      self.stop_idx = len(self.dirs)
      print('Loaded test data(%d - %d objects)' % (self.start_idx, self.stop_idx))
 

    self.seed_set = False

  def get_sequence(self):
    t = self.seq_len
    idx = np.random.randint(self.start_idx, self.stop_idx)
    obj_dir = self.dirs[idx]
    elevation = self.elevations[random.randint(0, len(self.elevations)-1)]
    dname = '%s/%s/%s' % (self.data_root, obj_dir, elevation)
    
    st = random.randint(1, 36-t)
    seq = [] 
    for i in range(st, st+t):
      fname = '%s/p200-64_%d.png' % (dname, i)
      im = misc.imread(fname)/255.
      seq.append(im)
    return np.array(seq)

  def __getitem__(self, index):
    if not self.seed_set:
      self.seed_set = True
      random.seed(index)
      np.random.seed(index)
      #torch.manual_seed(index)
    return torch.from_numpy(self.get_sequence())

  def __len__(self):
    return len(self.dirs)*36*5

