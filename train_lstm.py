import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--model_path', default='', help='path to drnet model')
parser.add_argument('--data_root', default='', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--data', default='moving_mnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=10, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--data_threads', type=int, default=5, help='number of parallel data loading threads')
parser.add_argument('--data_type', default='sequence', help='speed up data loading for drnet training')


opt = parser.parse_args()
name = 'rnn_size=%d-rnn_layers=%d-n_past=%d-n_future=%d-lr=%.4f-normalize=%s' % (opt.rnn_size, opt.rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.normalize)
opt.log_dir = '%s/lstm/%s' % (opt.model_path, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
opt.max_step = opt.n_past+opt.n_future

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
checkpoint = torch.load('%s/model.pth' % opt.model_path)
netD = checkpoint['netD']
netEP = checkpoint['netEP']
netEC = checkpoint['netEC']
netD.train()
netEP.train()
netEC.train()
drnet_opt = checkpoint['opt']
opt.pose_dim = drnet_opt.pose_dim
opt.content_dim = drnet_opt.content_dim
opt.image_width = drnet_opt.image_width
opt.dataset = drnet_opt.dataset
opt.data_root = drnet_opt.data_root

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

import models.lstm as models
lstm = models.lstm(opt.pose_dim+opt.content_dim, opt.pose_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size, opt.normalize)

lstm.apply(utils.init_weights)

optimizer = opt.optimizer(lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()

# --------- transfer to gpu ------------------------------------
lstm.cuda()
netEP.cuda()
netEC.cuda()
netD.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------
def plot_gen(x, epoch):
    # get fixed content vector from last ground truth frame
    h_c = netEC(x[opt.n_past-1])
    if type(h_c) is tuple:
        vec_h_c = h_c[0].detach()
    else:
        vec_h_c = h_c.detach()

    lstm.hidden = lstm.init_hidden()
    gen_seq = []
    h_p = netEP(x[0]).detach()
    gen_seq.append(x[0])
    for i in range(1, opt.n_past+opt.n_future):
        if i < opt.n_past:
            lstm(torch.cat([h_p, vec_h_c], 1))
            h_p =netEP(x[i]).detach()
            gen_seq.append(x[i])
        else:
            h_p = lstm(torch.cat([h_p, vec_h_c], 1))
            pred_x = netD([h_c, h_p]).detach()
            gen_seq.append(pred_x)

    to_plot = []
    nrow = 10
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/gen_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)


def plot_rec(x, epoch):
    # get fixed content vector from last ground truth frame
    h_c = netEC(x[opt.n_past-1])
    if type(h_c) is tuple:
        vec_h_c = h_c[0].detach()
    else:
        vec_h_c = h_c.detach()

    lstm.hidden = lstm.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    for i in range(1, opt.n_past+opt.n_future):
        h_p = netEP(x[i-1]).detach()
        h_pred = lstm(torch.cat([h_p, vec_h_c], 1))
        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            pred_x = netD([h_c, h_pred]).detach()
            gen_seq.append(pred_x)

    to_plot = []
    nrow = 10
    for i in range(nrow):
        # ground truth
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(x[t][i]) 
        to_plot.append(row)
        # gen
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

# --------- training funtions ------------------------------------
def train(x):
    lstm.zero_grad()

    # initialize the hidden state.
    lstm.hidden = lstm.init_hidden()

    # get fixed content vector from last ground truth frame
    h_c = netEC(x[opt.n_past-1])
    if type(h_c) is tuple:
        h_c = h_c[0].detach()
    else:
        h_c = h_c.detach()
    # get sequence of pose vectors
    h_p = [netEP(x[i]).detach() for i in range(opt.n_past+opt.n_future)]

    mse = 0
    for i in range(1, opt.n_past+opt.n_future):
        pose_pred = lstm(torch.cat([h_p[i-1], h_c], 1)) 
        #if i >= opt.n_past:
        mse += mse_criterion(pose_pred, h_p[i])
    mse.backward()

    optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future)

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    lstm.train()
    epoch_loss = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        # train lstm 
        loss = train(x)
        epoch_loss += loss


    progress.finish()
    utils.clear_progressbar()

    lstm.eval()
    # plot some stuff
    x = next(testing_batch_generator)
    plot_gen(x, epoch)
    plot_rec(x, epoch)

    print('[%02d] mse loss: %.6f (%d)' % (epoch, epoch_loss/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # save the model
    torch.save({
        'lstm': lstm,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
        
