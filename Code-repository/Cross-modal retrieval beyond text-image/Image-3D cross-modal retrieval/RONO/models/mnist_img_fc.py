import torch.nn as nn
import torch.nn.functional as F
class Img_FC(nn.Module):
    def __init__(self):
        super(Img_FC,self).__init__()
        self.conv1 = nn.Conv2d(1,96,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(96,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*7*7,1024)#两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
#         self.dp = nn.Dropout(p=0.5)
    def forward(self,x):
        x= x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7* 7)#将数据平整为一维的 
        x = F.relu(self.fc1(x))
#         x = self.fc3(x)
#         self.dp(x)
        x = F.relu(self.fc2(x))   
        x = self.fc3(x)  
#         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x

# class Img_FC(nn.Module):

#     def __init__(self, pre_trained = None):
#         super(Img_FC, self).__init__()

#         # self.linear1 = nn.Linear(512, 512, bias=False)
#         # self.bn6 = nn.BatchNorm1d(512)
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1024, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True)
#         )

#     def forward(self, img_feat):
#        # img_feat = F.relu(self.bn6(self.linear1(img_feat1)))
#         #img_feat_v = F.relu(self.bn6(self.linear1(img_feat2)))
#         #final_feat = 0.5*(img_feat + img_feat_v)
#         output = self.classifier(img_feat)
#         return output
        
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
        
#         self.linear1 = nn.Linear(512, 512, bias=False)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout(p=0.5)
#         self.linear2 = nn.Linear(512, 256)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout(p=0.5)
#         self.linear3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = F.relu(self.bn6(self.linear1(x)))
#         x = self.dp1(x)
#         x = F.relu(self.bn7(self.linear2(x)))
#         return x

class HeadNet(nn.Module):

    def __init__(self, num_classes):
        super(HeadNet, self).__init__()
        self.num_classes=num_classes
	    #shared head for all feature encoders
        self.head = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])

    def forward(self, img_feat, pt_feat):

        img_pred = self.head(img_feat)
        pt_pred = self.head(pt_feat)
        vis_img_feat = self.head[0](img_feat)
        vis_pt_feat = self.head[0](pt_feat)
        return img_pred, pt_pred, vis_img_feat, vis_pt_feat