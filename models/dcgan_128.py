import torch
import torch.nn as nn

class dcgan_conv(nn.Module):
  def __init__(self, nin, nout):
    super(dcgan_conv, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(nin, nout, 4, 2, 1),
      nn.BatchNorm2d(nout),
      nn.LeakyReLU(0.2, inplace=True),
      )

  def forward(self, input):
    return self.main(input)


class dcgan_upconv(nn.Module):
  def __init__(self, nin, nout):
    super(dcgan_upconv, self).__init__()
    self.main = nn.Sequential(
      nn.ConvTranspose2d(nin, nout, 4, 2, 1),
      nn.BatchNorm2d(nout),
      nn.LeakyReLU(0.2, inplace=True),
      )

  def forward(self, input):
    return self.main(input)

class pose_encoder(nn.Module):
  def __init__(self, pose_dim, nc=1, nf=64):
    super(pose_encoder, self).__init__()
    self.main = nn.Sequential(
      # input is (nc) x 128 x 128
      dcgan_conv(nc, nf),
      # state size. (nf) x 64 x 64
      dcgan_conv(nf, nf * 2),
      # state size. (nf*2) x 32 x 32
      dcgan_conv(nf * 2, nf * 4),
      # state size. (nf*4) x 16 x 16
      dcgan_conv(nf * 4, nf * 8),
      # state size. (nf*8) x 8 x 8
      dcgan_conv(nf * 8, nf * 8),
      # state size. (nf*8) x 4 x 4
      nn.Conv2d(nf * 8, pose_dim, 4, 1, 0),
      nn.BatchNorm2d(pose_dim),
      nn.Tanh()
      )

  def forward(self, input):
    return self.main(input)

class ind_pose_encoder(nn.Module):
  def __init__(self, n_pose, pose_dim, sep_path=False, nc=1, nf=32):
    super(ind_pose_encoder, self).__init__()
    self.n_pose = n_pose
    self.pose_dim = pose_dim
    self.sep = sep_path
    if self.sep:
      nets = []
      for i in range(n_pose):
        nets.append(nn.Sequential(
        # input is (nc) x 128 x 128
        dcgan_conv(nc, nf),
        # state size. (nf) x 64 x 64
        dcgan_conv(nf, nf * 2),
        # state size. (nf*2) x 32 x 32
        dcgan_conv(nf * 2, nf * 4),
        # state size. (nf*4) x 16 x 16
        dcgan_conv(nf * 4, nf * 8),
        # state size. (nf*8) x 8 x 8
        dcgan_conv(nf * 8, nf * 8),
        # state size. (nf*8) x 4 x 4
        nn.Conv2d(nf * 8, pose_dim, 4, 1, 0),
        nn.BatchNorm2d(pose_dim),
        nn.Tanh()
          ))
      self.nets = nn.ModuleList(nets)
    else:
      self.net = nn.Sequential(
        # input is (nc) x 128 x 128
        dcgan_conv(nc, nf),
        # state size. (nf) x 64 x 64
        dcgan_conv(nf, nf * 2),
        # state size. (nf*2) x 32 x 32
        dcgan_conv(nf * 2, nf * 4),
        # state size. (nf*4) x 16 x 16
        dcgan_conv(nf * 4, nf * 8),
        # state size. (nf*8) x 8 x 8
        dcgan_conv(nf * 8, nf * 8),
        # state size. (nf*8) x 4 x 4
        nn.Conv2d(nf * 8, pose_dim*n_pose, 4, 1, 0),
        nn.BatchNorm2d(pose_dim*n_pose),
        nn.Tanh()
          )

  def forward(self, input):
    if self.sep:
      output = []
      for i in range(self.n_pose):
        output.append(self.nets[i](input))
    else:
      combined = self.net(input)
      output = []
      for i in range(0, self.n_pose*self.pose_dim, self.pose_dim):
        output.append(combined[:, i:i+self.pose_dim])
    return output 

class content_encoder(nn.Module):
  def __init__(self, content_dim, nc=1, nf=64):
    super(content_encoder, self).__init__()
    self.main = nn.Sequential(
      # input is (nc) x 128 x 128
      dcgan_conv(nc, nf),
      # state size. (nf) x 64 x 64
      dcgan_conv(nf, nf * 2),
      # state size. (nf*2) x 32 x 32
      dcgan_conv(nf * 2, nf * 4),
      # state size. (nf*4) x 16 x 16
      dcgan_conv(nf * 4, nf * 8),
      # state size. (nf*8) x 8 x 8
      dcgan_conv(nf * 8, nf * 8),
      # state size. (nf*8) x 4 x 4
      nn.Conv2d(nf * 8, content_dim, 4, 1, 0),
      nn.BatchNorm2d(content_dim),
      nn.Tanh()
      )

  def forward(self, input):
    return self.main(input)

class ind_content_encoder(nn.Module):
  def __init__(self, n_content, content_dim, nc=1, nf=64):
    super(ind_content_encoder, self).__init__()
    self.n_content = n_content
    nets = []
    for i in range(n_content):
      nets.append(nn.Sequential(
        # input is (nc) x 128 x 128
        dcgan_conv(nc, nf),
        # state size. (nf) x 64 x 64
        dcgan_conv(nf, nf * 2),
        # state size. (nf*2) x 32 x 32
        dcgan_conv(nf * 2, nf * 4),
        # state size. (nf*4) x 16 x 16
        dcgan_conv(nf * 4, nf * 8),
        # state size. (nf*8) x 8 x 8
        dcgan_conv(nf * 8, nf * 8),
        # state size. (nf*8) x 4 x 4
        nn.Conv2d(nf * 8, content_dim, 4, 1, 0),
        nn.BatchNorm2d(content_dim),
        nn.Tanh()
        ))
    self.nets = nn.ModuleList(nets)

  def forward(self, input):
    output = []
    for i in range(self.n_content):
      output.append(self.nets[i](input))
    return output

class decoder(nn.Module):
  def __init__(self, content_dim, pose_dim, nc=1, nf=64):
    super(decoder, self).__init__()
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(content_dim+pose_dim, nf * 8, 4, 1, 0),
      nn.BatchNorm2d(nf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (nf*8) x 4 x 4
      dcgan_upconv(nf * 8, nf * 8),
      # state size. (nf*8) x 8 x 8
      dcgan_upconv(nf * 8, nf * 4),
      # state size. (nf*4) x 16 x 16
      dcgan_upconv(nf * 4, nf * 2),
      # state size. (nf*2) x 32 x 32
      dcgan_upconv(nf * 2, nf),
      # state size. (nf) x 64 x 64
      nn.ConvTranspose2d(nf, nc, 4, 2, 1),
      nn.Sigmoid()
      # state size. (nc) x 128 x 128
    )

  def forward(self, input):
    content, pose = input
    if type(content) == list:
      content = torch.cat(content, 1)
    if type(pose) == list:
      pose = torch.cat(pose, 1)
    output = self.main(torch.cat([content, pose], 1))
    return output

class scene_discriminator(nn.Module):
  def __init__(self, pose_dim, nf=256):
    super(scene_discriminator, self).__init__()
    self.pose_dim = pose_dim
    self.main = nn.Sequential(
      nn.Linear(pose_dim*2, nf),
      nn.ReLU(True),
      nn.Linear(nf, nf),
      nn.ReLU(True),
      nn.Linear(nf, 1),
      nn.Sigmoid(),
    )

  def forward(self, input):
    output = self.main(torch.cat(input, 1).view(-1, self.pose_dim*2))
    return output

class independent_discriminator(nn.Module):
  def __init__(self, pose_dim, nf=256):
    super(independent_discriminator, self).__init__()
    self.pose_dim = pose_dim
    self.main = nn.Sequential(
      nn.Linear(pose_dim, nf),
      nn.ReLU(True),
      nn.Linear(nf, nf),
      nn.ReLU(True),
      nn.Linear(nf, 1),
      nn.Sigmoid(),
    )

  def forward(self, input):
    return self.main(torch.cat(input, 1).view(-1, self.pose_dim))
    
