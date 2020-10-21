import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt

def sample():
    x = torch.randn(1000, 2, dtype=torch.float)
    #print(x)
    y = torch.zeros(1000, 1, dtype=torch.float)
    #print(y)
    for i in range(1000):
        a = torch.sum(torch.pow(x[i], 2))
        if a > 1.0:
            y[i][0] = 1.0
    #print(y)
    x, y = Variable(x), Variable(y)
    return (x, y)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        #self.middle = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x):
        x = f.relu(self.hidden(x))
        #x = f.relu(self.middle(x))
        x = self.predict(x)
        return x

net = Net(2, 10, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()
#loss_func = torch.nn.CrossEntropyLoss()
for t in range(10000):
    x,y=sample()
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    a = torch.abs(y-prediction)>0.5
    count=0
    for i in a:
        #print( i )
        if i==True:
            count+=1
    print("count=",count)

    #print ("diff=", torch.abs(y-prediction)>0.5)
    print("loss=", loss.data)
print(net.parameters())

