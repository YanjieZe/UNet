from data_loader2d import SunrgbdImageDataset
from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import visdom
import numpy as np

def collect_function(batchs):
    """
    Get a batch look as what our need 
    """
    images = None
    seg_labels = None
    for batch in batchs:
        if images is None:
            images = batch['image'].unsqueeze(dim=0)
            seg_labels = batch['seg_label'].unsqueeze(dim=0)
            continue
        image = batch['image'].unsqueeze(dim=0)
        seg_label = batch['seg_label'].unsqueeze(dim=0)
        images = torch.cat([images, image], dim=0)
        seg_labels = torch.cat([seg_labels, seg_label], dim=0)

    return {'image': images, 'seg_label': seg_labels}
        

def train(epoch_num=50, vis=True):

    device = torch.device('cuda:1')
    net = UNet().to(device)
    dataset = SunrgbdImageDataset()
    data_loader = torchdata.DataLoader(dataset, batch_size=2, collate_fn=collect_function)
    optimzer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.999)
    loss_function = nn.BCELoss()
    
    if visdom:
        server = visdom.Visdom(env='Unet Segmentation on SUNRGBD')
    for epoch in range(epoch_num):
        for idx,batch in enumerate(data_loader):
            img = batch['image'].to(device)
            seg_label = batch['seg_label'].to(device)
            
            seg_pred = net(img)
            loss = loss_function(seg_pred, seg_label)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            if idx%10==0:
                if vis:
                    server.line(X=np.array([idx]), Y=np.array([loss.item()]),update='append', win='SUNGRBD')
                else:
                    print("epoch:{} idx:{} loss:{}".format(epoch, idx, loss))
            

if __name__=='__main__':
    train()