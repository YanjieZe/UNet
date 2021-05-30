import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import cv2
import hydra
import numpy as np
import os
import visdom
from utils.sunrgbd_utils import read_sunrgbd_label
import scipy.io as sio
import PIL.Image as pil

seg_list37 = {
  1: 'wall',
  2: 'floor',
  3:'cabinet',
  4: 'bed',
  5: 'chair',
  6:'sofa',
  7: 'table',
  8: 'door',
  9: 'window',
  10: 'bookshelf',
  11:'picture',
   12:'counter',
   13:'blinds',
   14:'desk',
   15:'shelves',
   16:'curtain',
   17:'dresser',
   18:'pillow',
   19:'mirror',
   20:'floor_mat',
   21:'clothes',
   22:'ceiling',
   23:'books',
   24:'fridge',
   25:'tv',
   26:'paper',
   27:'towel',
   28:'shower_curtain',
   29:'box',
   30:'whiteboard',
   31:'person',
   32:'night_stand',
   33:'toilet',
   34:'sink',
   35:'lamp',
   36:'bathtub',
   37:'bag',
}

seg_list13 = {
    1: 'Bed',
	2:'Books',
3	:'Ceiling',
4	:'Chair',
5	:'Floor',
6:'Furniture',
7:'Objects',
8	:'Picture',
9	:'Sofa',
10	:'Table',
11	:'TV',
12	:'Wall',
13	:'Window',
}

class SunrgbdImageDataset(torchdata.Dataset):
    """
    TODO: Depth Image not used currently
    """
    def __init__(self, raw_datapath='/home/neil/disk/sunrgbd_trainval', 
            split_set='train', 
            class_name='lamp'):
        super(SunrgbdImageDataset, self).__init__()
        self.raw_datapath = raw_datapath
        self.class_name = class_name
        
        self.class_idx = None
        
        i=1
        while(i<=37):   
            if seg_list37[i]==self.class_name:
                self.class_idx = i
                break
            i += 1
    
        self.use_v1 = False
        self.image_path = os.path.join(self.raw_datapath, 'image')
        self.depth_path = os.path.join(self.raw_datapath, 'depth')
        self.image_names = open(os.path.join(self.raw_datapath, 'data_list', split_set,'{}.txt'.format(class_name))).read().splitlines()
        
       
    def augmentaion(self, image):
        """
        TODO: add more augmentation methods
        """
        # resize
        resize =transforms.Resize(size=(512,512))
        image = resize(image)

        # gray
        trans_gray = transforms.Grayscale()
        image = trans_gray(image)
        return image
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):      
        image_name = '{:06d}'.format(int(self.image_names[index]))
        image_path = os.path.join(self.image_path, image_name+'.jpg')
        depth_path = os.path.join(self.depth_path, image_name+'.mat')

        # RGB image
        image = pil.open(image_path)
        # augment
        image = self.augmentaion(image)
        # to tensor
        totensor = transforms.ToTensor()
        image = totensor(image)

        # Depth image (to be used)
        depth_image = sio.loadmat(depth_path)['instance']
        depth_image = torch.tensor(depth_image)

        objects =  read_sunrgbd_label(os.path.join(self.raw_datapath, 'label_v1' if self.use_v1 else 'label', '{}.txt'.format(image_name)))
        objects = [obj for obj in objects if obj.classname == self.class_name]
        bbox2d = [ obj.box2d for obj in objects] # (xmin, ymin, xmax, ymax)
       
        # segmentation label
        seg_label = sio.loadmat(os.path.join(self.raw_datapath, 'seg_label', image_name+'.mat'))['instance']
        seg_label = torch.tensor(seg_label)
        mask = seg_label==self.class_idx
        seg_label[:] = 0
        seg_label[mask] = 1
        
       

        return {'image_name':image_name,'image':image, 'seg_label':seg_label, 'bbox2d':bbox2d, 'depth_image':depth_image}


    def show_image(self, image):
        #server = visdom.Visdom(env='Sun RGBD Image')
        cv2.imwrite("./demo/image.jpg", image)
        

if __name__=='__main__':
    dataset = SunrgbdImageDataset(split_set='train')
    #print(dataset.image_names)

    print(dataset[2]['image'].shape)

