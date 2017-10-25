import random
import os
import numpy as np
import socket
import torch
from scipy import misc

hostname = socket.gethostname()
if hostname == 'ned' or hostname == 'zaan': 
  path = '/speedy/data/suncg/single_object/chair'
else:
  path = '/misc/vlgscratch3/FergusGroup/denton/suncg/factorized_datagen/chair//'

class SUNCG(object):

  def __init__(self, train, seq_length = 20, image_size=64):
    self.seq_length = seq_length
    self.image_size = image_size 

    self.elevations = ['elevation_1.25', 'elevation_-1.00', 'elevation_0.50', 'elevation_2.00', 'elevation_-0.25']
    self.dirs = os.listdir(path)
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
    t = self.seq_length
    idx = np.random.randint(self.start_idx, self.stop_idx)
    obj_dir = self.dirs[idx]
    elevation = self.elevations[random.randint(0, len(self.elevations)-1)]
    dname = '%s/%s/%s' % (path, obj_dir, elevation)
    
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

class DualSUNCG(object):

  def __init__(self, seq_length = 20, image_size=64):
    self.seq_length = seq_length
    self.image_size = image_size 

    self.elevations = ['elevation_1.25', 'elevation_-1.00', 'elevation_0.50', 'elevation_2.00', 'elevation_-0.25']
    self.dirs = os.listdir(path)
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
    t = self.seq_length
    seq = [] 

    # object 2
    idx = np.random.randint(self.start_idx, self.stop_idx)
    obj_dir = self.dirs[idx]
    elevation = self.elevations[random.randint(0, len(self.elevations)-1)]
    dname = '%s/%s/%s' % (path, obj_dir, elevation)
    st = random.randint(1, 36-t)
    for i in range(st, st+t):
      im = np.ones((128, 128, 3))
      fname = '%s/p200-64_%d.png' % (dname, i)
      im[:64, :64] = misc.imread(fname)/255.
      seq.append(im)

    # object 2
    idx = np.random.randint(self.start_idx, self.stop_idx)
    obj_dir = self.dirs[idx]
    elevation = self.elevations[random.randint(0, len(self.elevations)-1)]
    dname = '%s/%s/%s' % (path, obj_dir, elevation)
    st = random.randint(1, 36-t)
    k=0
    for i in range(st, st+t):
      fname = '%s/p200-64_%d.png' % (dname, i)
      seq[k][64:, 64:] = misc.imread(fname)/255.
      k+=1
    
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
