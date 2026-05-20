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

gru= nn.GRU(input_size=96,hidden_size=32,batch_first=True)
x=out3.permute(0,2,1)
output,hn= gru(x)
output.shape

gru1= nn.GRU(input_size=96,hidden_size=32,batch_first=True)
gru2= nn.GRU(input_size=32,hidden_size=32,batch_first=True)
gru_out1, _ = gru1(x)
gru_out2, _ = gru2(gru_out1)
gru_out=torch.cat((gru_out1,gru_out2),dim=2)
gru_out.shape
# gru_out1.shape

gru3= nn.GRU(input_size=64,hidden_size=32,batch_first=True)
gru_out3, _ = gru3(gru_out)
gru_out3.shape

gru_out=torch.cat((gru_out1,gru_out2,gru_out3),dim=2)
gru_out.shape

gru4= nn.GRU(input_size=96,hidden_size=32,batch_first=True)
gru_out4, _ = gru4(gru_out)
gru_out4.shape

linear= nn.Linear(in_features=1875, out_features=1)
linear_out= linear(gru_out.permute(0,2,1))
linear_out.shape

gru4= nn.GRU(input_size=96,hidden_size=32,batch_first=True)
gru_out4, _ = gru4(linear_out.permute(0,2,1))
gru_out4.shape