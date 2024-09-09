import torch.nn as nn
from DWT import DWT
class Block(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, wavelet='haar'):
        super(Block, self).__init__()
        
        # Define the Conv2d layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Define ReLU activation
        self.relu = nn.ReLU(inplace=True)
        self.dwt_layer = DWT()
    
    def forward(self, x):
        # Apply 3 Conv2d layers with ReLU activations
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #Apply DWT layer
        x = self.dwt_layer(x)
        
        return x