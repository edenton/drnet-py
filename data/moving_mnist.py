import socket
import numpy as np
from torchvision import datasets, transforms

class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.seed_is_set = False # multi threaded loading

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      3),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 4)
            dy = np.random.randint(-4, 4)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    dy = -dy
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    dy = -dy
                    
                if sx < 0:
                    sx = 0 
                    dx = -dx
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    dx = -dx
                   
                x[t, sy:sy+32, sx:sx+32, n] = np.copy(digit.numpy())
                sy += dy
                sx += dx
        # pick on digit to be in front
        front = np.random.randint(self.num_digits)
        for cc in range(self.num_digits):
            if cc != front:
                x[:, :, :, cc][x[:, :, :, front] > 0] = 0
        return x

