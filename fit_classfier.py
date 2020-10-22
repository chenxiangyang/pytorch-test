import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def sample():
    x = torch.randn(1000, 2, dtype=torch.float)*3.0
    #print(x)
    y = torch.zeros(1000, 1, dtype=torch.float)
    #print(y)
    for i in range(1000):
        a = torch.sum(torch.pow(x[i], 2.0))
        #b = torch.sum(torch.pow(x[i]-torch.tensor([2.0,2.0]), 2.0))
        #if a > 1.0 and b > 2.0:
        #    y[i][0] = 1.0
        #else:
        #    y[i][0] = 0.0
        y[i][0] = torch.sin(3.0 * torch.sqrt(torch.sum(torch.pow(x[i], 2.0))))
    
    x, y = Variable(x), Variable(y)
    return (x, y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.middle = torch.nn.Linear(n_hidden, n_hidden)
        self.middle1 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = f.relu(self.middle(x))
        x = f.relu(self.middle1(x))
        x = self.predict(x)
        return x

net = Net(2, 100, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()


def show_sample():
    figure = plt.figure()
    ax = Axes3D(figure)
    #plt.ion() 
    #plt.clf()
    X = np.arange(-3,3,0.05)
    Y = np.arange(-3,3,0.05)
    X,Y = np.meshgrid(X,Y)
    x = X.reshape(1,-1)
    y = Y.reshape(1,-1)

    _z = np.concatenate((x,y),0)
    _z = _z.transpose((1,0))
    _z = torch.from_numpy(_z).float()

    Z = net(_z)
    Z = Z.detach().numpy()
    Z = Z.reshape(-1)

    Z = Z.reshape(X.shape)

    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
    
    plt.pause(1)
    plt.show()

for t in range(20000):
    x,y=sample()
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    a = torch.abs(y-prediction)>0.5
    count=0
    for i in a:
        if i==True:
            count+=1
    print("count=",count)
    print("loss=", loss.data)

show_sample();

print(net.parameters())

