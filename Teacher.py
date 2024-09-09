import torch.nn as nn
class Teacher(nn.Module):
    def __init__(self, resnet18):
        super(Teacher, self).__init__()
        
        # Initialize the two models
        self.resnet18 = resnet18
               
    def forward(self, x):
        
        # Get predictions from ResNet18
        resnet_pred = self.resnet18(x)
        
        return resnet_pred
