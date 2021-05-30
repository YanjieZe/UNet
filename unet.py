import torch
import torch.nn as nn
import numpy as np
# from data_loader2d import SunrgbdImageDataset 

class UNet(nn.Module):
    """
    UNet for segmentation, totally implemented by Yanjie Ze.
    """

    def __init__(self, input_width=512):
        super(UNet,self).__init__()

        # down
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.downsample3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.downsample4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        )

        # up
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024,512,kernel_size=3,padding=1)
        )
        
        self.block6 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,256,kernel_size=3,padding=1)
        )
        
        self.block7 = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128,kernel_size=3,padding=1)
        )
        
        self.block8 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,kernel_size=3,padding=1)
        )
        
        self.block9 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        # full convolution
        self.fullconv = nn.Conv2d(64,1,kernel_size=1)
        


    def forward(self, image):

        ### down ###

        #1
        block1_out = self.block1(image)# bs * 64 * 512 * 512

        #2
        tmp = self.downsample1(block1_out)
        block2_out = self.block2(tmp) # bs * 128 * 256 * 256

        #3
        tmp = self.downsample2(block2_out)
        block3_out = self.block3(tmp) # bs * 256 * 128 * 128
        
        #4
        tmp = self.downsample3(block3_out)
        block4_out = self.block4(tmp) # bs * 512 * 64 * 64

        #5
        tmp = self.downsample4(block4_out)
        block5_out = self.block5(tmp) # bs * 1024 * 32 * 32
        
        ### up ###

        #1
        up1 = self.up1(block5_out)
        up1 = torch.cat([up1, block4_out], dim=1)
        up1 = self.block6(up1) # bs * 512 * 64 * 64

        #2
        up2 = self.up2(up1)
        up2 = torch.cat([up2, block3_out], dim=1)
        up2 = self.block7(up2) # bs * 256 * 128 * 128

        #3
        up3 = self.up3(up2)
        up3 = torch.cat([up3, block2_out], dim=1)
        up3 = self.block8(up3)

        #4
        up4 = self.up4(up3)
        up4 = torch.cat([up4, block1_out], dim=1)
        up4 = self.block9(up4)
        
        ### final output ###

        seg = self.fullconv(up4)

        seg_pred = torch.sigmoid(seg)
        return seg_pred

if __name__=='__main__':
    net = UNet(512)
   
    image = torch.zeros(8,1,512,512)

    print(net(image).shape)

