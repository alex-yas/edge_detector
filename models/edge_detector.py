import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetLikeEdgeDetector(nn.Module):
    """ Model for edge detecting. Consists of simple encoder and decoder 
    with skip connections.
    """
    def __init__(self):
        super(UnetLikeEdgeDetector, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.upconv1 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.final = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))        
        x = self.pool(x1)       
        x2 = F.relu(self.conv2(x))
        x = self.pool(x2)
        x = F.relu(self.conv3(x))
        
        x = self.upsample(x)        
        x = F.relu(self.upconv1(torch.cat((x,x2), dim=1)))        
        x = self.upsample(x)
        x = F.relu(self.upconv2(torch.cat((x,x1), dim=1)))
        
        x = self.final(x)
        return x