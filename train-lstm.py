

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.softmax_func=nn.Softmax(dim=1)

    def forward(self, input_data):
        out,(h_t, c_t)=self.lstm(input_data)
        print("aaa",out.shape)
        out=self.fc(out[:,-1,:])
        print("bbb",out.shape)
        #out = self.softmax_func(out)

        return out

def get_input():
    rand_nums = np.floor(np.random.rand(10)*99)
    sorted_nums = np.sort(rand_nums)
    index_nums = np.zeros(20, dtype=np.int32)

    input_data = np.zeros( (20, 1, 100) ,dtype=np.float32)
    output_data = np.zeros( (20, 100), dtype=np.long)
    for i in range(10):
        input_data[i][0][int(rand_nums[i])] = 1.0
        output_data[i+10][int(sorted_nums[i])] = 1
        index_nums[i+10] = sorted_nums[i]
        index_nums[i] = 99
    input = Variable(torch.from_numpy(input_data).float())
    output = Variable(torch.from_numpy(index_nums).long())
    
    return input,output

def train():
    lstm = LSTM(100, 100, 5)
    optimizer = Adam(lstm.parameters(),lr=0.001)
    loss_fn=torch.nn.CrossEntropyLoss()

    for i in range(10000000):
        input,labels=get_input()
        output = lstm.forward(input)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("output:", torch.argmax(output, axis=1))
        #print("output:", output)
        print("labels:", labels)
        #print("input:", input)
        print("loss=", loss.data)

train()






