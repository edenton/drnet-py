import random
import os
import numpy as np
import socket
import torch
from scipy import misc
from torch.utils.serialization import load_lua

def download_data(path):
    full_path = '%s/KTH/processed/' % path
    if not os.path.isdir(full_path):
        print('Downloading KTH dataset to %s' % path)
        import pdb; pdb.set_trace()
        url = 'http://www.cs.nyu.edu/~denton/datasets/kth.tar.gz'
        os.system('wget -O %s/kth_data.tar.gz %s' % (path, url))
        os.system('tar -xzvf %s/kth_data.tar.gz -C %s' % (path, path))
    return full_path

class KTH(object):

    def __init__(self, train, data_root, seq_len = 20, image_size=64):
        self.data_root = download_data(data_root)
        self.seq_len = seq_len
        self.image_size = image_size 
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

        self.dirs = os.listdir(self.data_root)
        if train:
            data_type = 'train'
            self.persons = list(range(1, 21))
        else:
            self.persons = list(range(21, 26))
            data_type = 'test'

        self.data= {}
        for c in self.classes:
            self.data[c] = load_lua('%s/%s/%s_meta%dx%d.t7' % (self.data_root, c, data_type, image_size, image_size))
     

        self.seed_set = False

    def get_sequence(self):
        t = self.seq_len
        while True: # skip seqeunces that are too short
            c_idx = np.random.randint(len(self.classes))
            c = self.classes[c_idx]
            vid_idx = np.random.randint(len(self.data[c]))
            vid = self.data[c][vid_idx]
            seq_idx = np.random.randint(len(vid['files']))
            if len(vid['files'][seq_idx]) - t >= 0:
                break
        dname = '%s/%s/%s' % (self.data_root, c, vid['vid'])
        st = random.randint(0, len(vid['files'][seq_idx])-t)
           

        seq = [] 
        for i in range(st, st+t):
            fname = '%s/%s' % (dname, vid['files'][seq_idx][i])
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

