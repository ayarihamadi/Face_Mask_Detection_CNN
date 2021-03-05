import torchvision
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, feat_dim = 1280, dim_output=2):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.dim_output = dim_output

        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)

        # # Fix Initial Layers
        for p in list(self.backbone.children())[:-1]:
            p.requires_grad = False

        # # get the structure until the Fully Connected Layer
        modules = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # Add new fully connected layers
        self.fc1 = nn.Linear(1280, 256) 
        self.fc2 = nn.Linear(256, 64) 
        self.fc3 = nn.Linear(64, 2) 
        self.avgpool= nn.AvgPool2d(7)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        batch_size = img.shape[0]
        out = self.backbone(img) # get the feature from the pre-trained resnet
        out = self.avgpool(out)
        out = self.dropout(self.relu(self.fc1(out.view(batch_size, -1))))
        out = self.dropout(self.relu(self.fc2(out))) 
        out = (self.sigmoid(self.fc3(out))) 
        return out
