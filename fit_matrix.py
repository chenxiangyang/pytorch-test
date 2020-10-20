
import torch

w=3000
h=2000
'''
M=torch.tensor([[0.6,0.8,0.2,0.6],
                [0.6,0.8,0.3,0.6],
                [0.6,0.8,0.2,0.9],
                [0.9,0.1,0.2,0.6],
                [0.1,0.6,0.1,0.1]], dtype=torch.float)
'''
M=torch.randn(h,w,dtype=torch.float)

def sample():
    x=torch.randn(w, 1, dtype=torch.float, requires_grad=True)
    y=torch.mm(M,x)
    return (x,y)

fitM=torch.randn(h, w, dtype=torch.float, requires_grad=True)

for i in range(200000):
    print("*********")
    s=sample()
    y=torch.mm(fitM, s[0])
    y.backward(s[1]-y)
    
    grad=fitM.grad*0.0001
    fitM = fitM.data + grad
    fitM.requires_grad_()
    #print("m =", fitM)
    #    usleep(100)

    print("diff=", torch.mean(abs(s[1]-y)).data)

print (M)
print (fitM)


    
