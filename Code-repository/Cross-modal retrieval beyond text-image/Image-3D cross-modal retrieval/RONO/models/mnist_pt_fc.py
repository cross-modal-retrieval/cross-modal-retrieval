
import torch.nn as nn
import torch.nn.functional as F

class Pt_fc(nn.Module):
    def __init__(self, args, output_channels=10):
        super(Pt_fc, self).__init__()
        self.args = args
        
        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        return x