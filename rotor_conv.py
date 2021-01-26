import torch
import torch.nn as nn

class RotorConvNet(nn.Module):
    def __init__(self,batch_size,classify=True):
        super(RotorConvNet, self).__init__()
        self.num_filters = 64
        self.in_channels = 1
        self.batch_size = batch_size
        self.classify = classify
        self.conv = nn.Conv2d(self.in_channels, self.num_filters, 3) 
        #self.bn = nn.BatchNorm2d(self.num_filters)
        self.relu = nn.ReLU()
        if self.classify:
            self.fc = nn.Linear(self.num_filters,2)
            self.sm = nn.Softmax()
        else:
            self.fc = nn.Linear(self.num_filters,1)

    def forward(self,x):
        batch_size = int(x.size()[0])
        #print(batch_size)
        #x = torch.reshape(x,(batch_size,self.in_channels,3,3))
        #print(x.shape)
        x = torch.unsqueeze(x,1)
        x = self.conv(x).squeeze()
        #x = self.bn(x)
        #x = self.relu(x).squeeze()
        #print(x.shape)
        #x = x.reshape(batch_size*self.num_filters)
        #print(x.shape)
        x = self.fc(x)
        if self.classify:
            x = self.sm(x)
        #x = (x+100)/(220) * 2.4 - 0.4
        #print(x.shape)
        return x #torch.cuda.FloatTensor(x)

    def reset_fc(self):
        if self.classify:
            self.fc = nn.Linear(self.num_filters,2)
            self.sm = nn.Softmax()
        else:
            self.fc = nn.Linear(self.num_filters,1)


class RotorFullyConnected(nn.Module):
    def __init__(self,batch_size):
        super(RotorFullyConnected, self).__init__()
        self.num_features = 8
        self.fc1 = nn.Linear(self.num_features,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,256)
        self.fc4 = nn.Linear(256,64)
        self.fc5 = nn.Linear(64,2)
        #self.sm = nn.Softmax(dim = 2)


    def forward(self,x):
        batch_size = int(x.size()[0])
        #print(batch_size)
        #x = torch.reshape(x,(batch_size,self.in_channels,3,3))
        #print(x.shape)
        x = torch.unsqueeze(x,1)
        #x = self.conv(x)
        #x = self.bn(x)
        #x = self.relu(x).squeeze()
        #print(x.shape)
        #x = x.reshape(batch_size*self.num_filters)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        #print("Before: {}".format(x))
        #print(x.shape)
        x = self.sm(x)
        #print("After: {}".format(x))
        #x = (x+100)/(220) * 2.4 - 0.4
        #print(x.shape)


        return x #torch.cuda.FloatTensor(x)

if __name__ == "__main__":
    model = RotorConvNet(16,classify=False)
    model.reset_fc()

