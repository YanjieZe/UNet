from data_loader2d import SunrgbdImageDataset
from unet import UNet
import torch
import torch.optim as optim
import torch.utils.data as torchdata

def train(epoch_num=20):
    net = UNet()
    dataset = SunrgbdImageDataset()
    data_loader = torchdata.DataLoader(dataset, batch_size=8,num_workers=8)
    optimzer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=0.999)

    for epoch in range(epoch_num):
        
        for idx,batch in enumerate(data_loader):
            img = batch['image']
            seg
