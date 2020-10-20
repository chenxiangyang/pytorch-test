import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class ClassfierNet(nn.Module):

    '''
    def __init__(self):
        super(ClassfierNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    '''

    '''
    def __init__(self):
        super(ClassfierNet, self).__init__()

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(12)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 8, 5)
        self.conv3 = nn.Conv2d(8, 12, 5)
        self.conv4 = nn.Conv2d(12, 16, 5)
        self.conv5 = nn.Conv2d(16, 16, 5)

        self.fc1 = nn.Linear(16*12*12, 20)
        self.fc2 = nn.Linear(20, 15) 
        self.fc3 = nn.Linear(15, 10)

    def forward(self, x):
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    '''
    '''
    def __init__(self):
        super(ClassfierNet, self).__init__()

        self.bn1   = nn.BatchNorm2d(3)
        self.bn1_2 = nn.BatchNorm2d(3)
        self.bn2   = nn.BatchNorm2d(3)
        self.bn2_shortcut = nn.BatchNorm2d(16)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.bn3   = nn.BatchNorm2d(32)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.bn3_3 = nn.BatchNorm2d(32)
        self.bn3_4 = nn.BatchNorm2d(32)
        self.bn4   = nn.BatchNorm2d(32)

        self.conv1   = nn.Conv2d(3, 3, 3)  #30
        self.conv1_2 = nn.Conv2d(3, 3, 3)  #28
        self.conv2   = nn.Conv2d(3, 16, 2, stride=2)  #14
        self.conv2_shortcut = nn.Conv2d(16, 32, 1)  #14
        self.conv2_1 = nn.Conv2d(16, 32, 3, padding=1)  #14
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)  #14
        self.conv3   = nn.Conv2d(32, 32, 2, stride=2)  #7
        self.conv3_1 = nn.Conv2d(32, 32, 3, padding=1)  #7
        self.conv3_2 = nn.Conv2d(32, 32, 3, padding=1)  #7
        self.conv3_3 = nn.Conv2d(32, 32, 3, padding=1)  #7
        self.conv3_4 = nn.Conv2d(32, 32, 3, padding=1)  #7
        self.conv4   = nn.Conv2d(32, 32, 3, stride=2)  #3
        self.conv5   = nn.Conv2d(32, 64, 3) #1
        self.conv6   = nn.Conv2d(64, 10, 1) #1

    def forward(self, x):
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv1_2(self.bn1_2(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        short_cut0 = x
        x = F.relu(self.conv2_1(self.bn2_1(x)))
        x = F.relu(self.conv2_2(self.bn2_2(x)))
        short_cut0 = F.relu(self.conv2_shortcut(self.bn2_shortcut(short_cut0)))
        x += short_cut0
        x = F.relu(self.conv3(self.bn3(x)))
        short_cut1=x
        x = F.relu(self.conv3_1(self.bn3_1(x)))
        x = F.relu(self.conv3_2(self.bn3_2(x)))
        x+=short_cut1
        short_cut2=x
        x = F.relu(self.conv3_3(self.bn3_3(x)))
        x = F.relu(self.conv3_4(self.bn3_4(x)))
        x+=short_cut2
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, self.num_flat_features(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    '''
    
    def __init__(self):
        super(ClassfierNet, self).__init__()

        self.bn1   = nn.BatchNorm2d(3)
        self.bn2   = nn.BatchNorm2d(64)
        self.bn3   = nn.BatchNorm2d(128)
        self.bn4   = nn.BatchNorm2d(128)
        self.bn5   = nn.BatchNorm2d(128)
        self.bn6   = nn.BatchNorm2d(128)
        self.bn7   = nn.BatchNorm2d(128)
        self.bn8   = nn.BatchNorm2d(128)
        self.bn9   = nn.BatchNorm2d(128)
        self.bn10   = nn.BatchNorm2d(128)
        self.bn11   = nn.BatchNorm2d(112)
        self.bn12   = nn.BatchNorm2d(112)
        self.bn13   = nn.BatchNorm2d(112)
        self.bn14   = nn.BatchNorm2d(112)
        self.bn15   = nn.BatchNorm2d(112)
        self.bn16   = nn.BatchNorm2d(124)
        self.bn17   = nn.BatchNorm2d(124)
        self.bn18   = nn.BatchNorm2d(124)
        self.bn19   = nn.BatchNorm2d(124)
        self.bn20   = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  #32
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)  #32
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv9 = nn.Conv2d(128, 128, 3, padding=1)  #32
        self.conv10 = nn.Conv2d(128, 112, 3, stride=2)  #15
        self.conv11 = nn.Conv2d(112,112, 3, padding=1)  #15
        self.conv12 = nn.Conv2d(112,112, 3, padding=1)  #15
        self.conv13 = nn.Conv2d(112,112, 3, padding=1)  #15
        self.conv14 = nn.Conv2d(112,112, 3, padding=1)  #15
        self.conv15 = nn.Conv2d(112,124, 3, stride=2)  #7
        self.conv16 = nn.Conv2d(124,124, 3, padding=1)  #7
        self.conv17 = nn.Conv2d(124,124, 3, padding=1)  #7
        self.conv18 = nn.Conv2d(124,124, 3, padding=1)  #7
        self.conv19 = nn.Conv2d(124,32, 3, stride=2)  #3
        self.conv20 = nn.Conv2d(32,10, 3)  #1

        self.conv_shortcut1 = nn.Conv2d(3, 128, 1)  #32
        self.bn_shortcut1 = nn.BatchNorm2d(3)

        self.conv_shortcut5 = nn.Conv2d(128, 112, 3, stride=2)  #15
        self.bn_shortcut5 = nn.BatchNorm2d(128)

        self.conv_shortcut8 = nn.Conv2d(112, 124, 3, stride=2)  #15 --> 7
        self.bn_shortcut8 = nn.BatchNorm2d(112)

        self.conv_shortcut10 = nn.Conv2d(124, 10, 7)  #7 --> 1
        self.bn_shortcut10 = nn.BatchNorm2d(124)

    def forward(self, x):
        short_cut1 = x
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        short_cut1=F.relu(self.conv_shortcut1(self.bn_shortcut1(short_cut1)))
        x+=short_cut1
        x=F.relu(x)
        
        short_cut2=x
        x = F.relu(self.conv3(self.bn3(x)))  
        x = F.relu(self.conv4(self.bn4(x)))
        x+=short_cut2
        
        x=F.relu(x)
        short_cut3=x
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.relu(self.conv6(self.bn6(x)))
        x+=short_cut3

        x=F.relu(x)
        
        short_cut4=x
        x = F.relu(self.conv7(self.bn7(x)))
        x = F.relu(self.conv8(self.bn8(x)))
        x+=short_cut4
        x=F.relu(x)        

        short_cut5=x
        x = F.relu(self.conv9(self.bn9(x)))
        x = F.relu(self.conv10(self.bn10(x)))
        short_cut5=F.relu(self.conv_shortcut5(self.bn_shortcut5(short_cut5)))
        x+=short_cut5

        x=F.relu(x)
        short_cut6=x
        x = F.relu(self.conv11(self.bn11(x)))
        x = F.relu(self.conv12(self.bn12(x)))
        x+=short_cut6

        x=F.relu(x)
        short_cut7=x
        x = F.relu(self.conv13(self.bn13(x)))
        x = F.relu(self.conv14(self.bn14(x)))
        x+=short_cut7

        x=F.relu(x)
        short_cut8=x
        x = F.relu(self.conv15(self.bn15(x)))
        x = F.relu(self.conv16(self.bn16(x)))
        short_cut8=F.relu(self.conv_shortcut8(self.bn_shortcut8(short_cut8)))
        x+=short_cut8

        x=F.relu(x)
        short_cut9=x
        x = F.relu(self.conv17(self.bn17(x)))
        x = F.relu(self.conv18(self.bn18(x)))
        x+=short_cut9
        x = F.relu(self.conv19(self.bn19(x)))
        x = F.relu(self.conv20(self.bn20(x)))

        x = x.view(-1, self.num_flat_features(x))
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

net = ClassfierNet()
print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

def samples():
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainloader,testloader,classes)


def train():
    trainloader,testloader,classes = samples()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.01)

    for epoch in range(100):
        running_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for i, data in enumerate(trainloader, 0):
            inputs,labels=data
            inputs, labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            #print(labels.shape)
            #print(i)
            for j in range(100):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1

            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                print("class_correct:", class_correct)
                print("class_total", class_total)
                print("p:",sum(class_correct)/sum(class_total))
                
                for i in range(10):
                    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

                class_correct = list(0. for i in range(10))
                class_total = list(0. for i in range(10))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for j, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            outputs=net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for j in range(100):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
        
        print("test class_correct:", class_correct)
        print("test class_total", class_total)
        print("test p:",sum(class_correct)/sum(class_total))

train()

#input=torch.randn(1, 1, 32, 32, dtype=torch.float)
#net(input)



