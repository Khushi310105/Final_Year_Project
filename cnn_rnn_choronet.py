import torch.nn as nn
import torch
input= torch.randn(3,22,15000)
input.shape


nn.Conv1d(in_channels=22,out_channels=33,kernel_size=2,stride=2,padding=0)

class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=2, padding=3)
        self.relu= nn.ReLU()
    def forward(self,x):
        x1= self.conv1(x)
        x2= self.conv2(x)
        x3= self.conv3(x)
        x= torch.cat((x1,x2,x3),dim=1)
        # x= self.relu(x)
        return x

block= Block(in_channels=input.shape[1])
out1=block(input)
out1.shape

block= Block(in_channels=out1.shape[1])
out2=block(out1)
out2.shape

block= Block(in_channels=out2.shape[1])
out3=block(out2)
out3.shape