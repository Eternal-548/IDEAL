import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_class, input_len):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=input_len, out_features=128) 
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.out = nn.Linear(in_features=32,  out_features=num_class)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)  
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.out(out)

        return out
    