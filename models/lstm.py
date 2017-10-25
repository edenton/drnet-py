import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return  (Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()), 
                 Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()))

    def forward(self, input):
        self.hidden = self.lstm(input.view(-1, self.input_size), self.hidden)
        output = self.output(self.hidden[0])
        return output
