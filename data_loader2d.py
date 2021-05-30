import torch
import torch.utils.data as torchdata
import torchvision
import cv2
import hydra
import numpy as np
import os
import visdom
from utils.sunrgbd_utils import read_sunrgbd_label
import scipy.io as sio

class SunrgbdImageDataset(torchdata.Dataset):
    def __init__(self, raw_datapath='/home/neil/disk/sunrgbd_trainval', 
            split_set='train', 
            class_name='computer'):
        super(SunrgbdImageDataset, self).__init__()
        self.raw_datapath = raw_datapath
        self.class_name = class_name
        self.use_v1 = False
        self.image_path = os.path.join(self.raw_datapath, 'image')
        self.depth_path = os.path.join(self.raw_datapath, 'depth')
        self.image_names = open(os.path.join(self.raw_datapath, 'data_list', split_set,'{}.txt'.format(class_name))).read().splitlines()
        
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if str(index) not in self.image_names:
            raise Exception("Data Fetch Error: Current image idx not in this class.")
        image_name = '{:06d}'.format(index)
        image_path = os.path.join(self.image_path, image_name+'.jpg')
        depth_path = os.path.join(self.depth_path, image_name+'.mat')

        # RGB image
        image = cv2.imread(image_path)
        
        # Depth image (to be used)
        depth_image = sio.loadmat(depth_path)['instance']
        
        objects =  read_sunrgbd_label(os.path.join(self.raw_datapath, 'label_v1' if self.use_v1 else 'label', '{}.txt'.format(image_name)))
        objects = [obj for obj in objects if obj.classname == self.class_name]
        bbox2d = [ obj.box2d for obj in objects] # (xmin, ymin, xmax, ymax)

        # segmentation label
        seg_label = sio.loadmat(os.path.join(self.raw_datapath, 'seg_label', image_name+'.mat'))['instance']
        
        #print(np.max(seg_label))
        return image_name, image

    def show_image(self, image):
        #server = visdom.Visdom(env='Sun RGBD Image')
        cv2.imwrite("./demo/image.jpg", image)
        

if __name__=='__main__':
    dataset = SunrgbdImageDataset()
    #print(dataset.image_names)

    dataset.show_image(dataset[7828][1])

