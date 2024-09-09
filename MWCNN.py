import torch
import torch.nn as nn
from Block import Block
class MWCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, wavelet='haar'):
        super(MWCNN, self).__init__()
        #with Dwt
        # Define two Block instances
        self.block1 = Block(in_channels=in_channels, out_channels=out_channels, wavelet=wavelet)
        self.block2 = Block(in_channels=4, out_channels=128, wavelet=wavelet)
        self.block3 = Block(in_channels=4, out_channels=256, wavelet=wavelet)
        self.block4 = Block(in_channels=4, out_channels=512, wavelet=wavelet)
        self.block5 = Block(in_channels=4, out_channels=1024, wavelet=wavelet)
        
        self.fc1 = nn.Linear(4 * 7 * 7, 512)  # Adjust size according to input image size
        self.fc2 = nn.Linear(512, 4)  # Assuming 10 classes for classification
        
    def forward(self, x):
        # Pass input through the first Block
        x = self.block1(x)
    
        x = self.block2(x)

        x = self.block3(x)
    
        x = self.block4(x)
    
        x = self.block5(x)
    

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
