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
    def __init__(self, pose_dim, nc=1, normalize=False):
        super(pose_encoder, self).__init__()
        nf = 64
        self.normalize = normalize
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                dcgan_conv(nc, nf),
                # state size. (nf) x 32 x 32
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 4 x 4
                nn.Conv2d(nf * 8, pose_dim, 4, 1, 0),
                nn.BatchNorm2d(pose_dim),
                nn.Tanh()
                )

    def forward(self, input):
        output = self.main(input)
        if self.normalize:
            return nn.functional.normalize(output, p=2)
        else:
            return output

class content_encoder(nn.Module):
    def __init__(self, content_dim, nc=1):
        super(content_encoder, self).__init__()
        nf = 64
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                dcgan_conv(nc, nf),
                # state size. (nf) x 32 x 32
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 4 x 4
                nn.Conv2d(nf * 8, content_dim, 4, 1, 0),
                nn.BatchNorm2d(content_dim),
                nn.Tanh()
                )

    def forward(self, input):
        return self.main(input)

class decoder(nn.Module):
    def __init__(self, content_dim, pose_dim, nc=1):
        super(decoder, self).__init__()
        nf = 64
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(content_dim+pose_dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (nf*8) x 4 x 4
                dcgan_upconv(nf * 8, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_upconv(nf * 4, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_upconv(nf * 2, nf),
                # state size. (nf) x 32 x 32
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        content, pose = input
        return self.main(torch.cat([content, pose], 1))

