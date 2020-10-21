
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#M=torch.randn(h,w,dtype=torch.float)

def sample():
    x=torch.randn(2, 1, dtype=torch.float)*1.0
    y=(x[0][0]*x[0][0]/16.0 + x[1][0]*x[1][0]/9.0)
    '''
    if y==True:
        y=1
    else:
        y=0
    '''
    return (x,y)

M0=torch.randn(100,2,dtype=torch.float, requires_grad=True)
M1=torch.randn(2,100,dtype=torch.float, requires_grad=True)
#M2=torch.randn(2,2,dtype=torch.float, requires_grad=True)
print(sample())

for i in range(200000):
    s=sample()
    #print("s[0]=", s[0])
    #print("transpose:", torch.transpose(s[0], 0, 1))
    y=torch.mm(M0, s[0])
    y=torch.relu(y)
    y=torch.mm(M1, y)
    #y=F.relu(y)
    #y=torch.mm(M2, y)
    #y=F.relu(y)
    y=torch.mm(torch.transpose(s[0], 0, 1), y)
    t=torch.randn(1,1,dtype=torch.float)
    t[0][0]=s[1]-y[0][0]
    y.backward(t)
    grad=M0.grad*0.0001
    M0 = M0.data + grad
    M0.requires_grad_()

    grad=M1.grad*0.0001
    M1 = M1.data + grad
    M1.requires_grad_()

    #grad=M2.grad*0.0001
    #M2 = M2.data + grad
    #M2.requires_grad_()

    #print(y.data,s[1])
    print("diff=",t[0][0].data)

print("M0=",M0)
#print("M1=",M1)
#print("M2=",M2)

