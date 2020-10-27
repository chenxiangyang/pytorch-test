import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def sample():
    x = torch.rand(100, 1, dtype=torch.float)*2.0-1.0
    #print(x)
    y = torch.zeros(100, 1, dtype=torch.float)
    for i in range(100):
        X=x[i][0]
        y[i][0]=0.8*X*X*X+2.0*X*X-1.0*X-0.5
    
    x, y = Variable(x), Variable(y)
    return (x, y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden1)
        self.middle = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
 
    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = f.relu(self.middle(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

for t in range(4000):
    x,y=sample()
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss=", loss.data)

print(net.parameters())
plt.figure()
x=np.linspace(-1.0,1.0,num=100)
y1=0.8*x*x*x+2.0*x*x-1.0*x-0.5
x1=torch.from_numpy(x).float()
x1=x1.reshape(100,1)
y2=net(x1)
y2=y2.reshape(100)
y2=y2.detach().numpy()
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()



