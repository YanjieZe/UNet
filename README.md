# UNet Segmenation on Sun-RGBD
Yanjie Ze, May&June 2021
# Model
UNet(the same as the model in that paper)
# data path
path = '/home/neil/disk/sunrgbd_trainval'

# Sun-RGBD Seg 37 list
```python
>> seg = load('seg37list.mat');
>> seg.seg37list
ans = 
{
   1] = wall
  [1,2] = floor
  [1,3] = cabinet
  [1,4] = bed
  [1,5] = chair
  [1,6] = sofa
  [1,7] = table
  [1,8] = door
  [1,9] = window
  [1,10] = bookshelf
  [1,11] = picture
  [1,12] = counter
  [1,13] = blinds
  [1,14] = desk
  [1,15] = shelves
  [1,16] = curtain
  [1,17] = dresser
  [1,18] = pillow
  [1,19] = mirror
  [1,20] = floor_mat
  [1,21] = clothes
  [1,22] = ceiling
  [1,23] = books
  [1,24] = fridge
  [1,25] = tv
  [1,26] = paper
  [1,27] = towel
  [1,28] = shower_curtain
  [1,29] = box
  [1,30] = whiteboard
  [1,31] = person
  [1,32] = night_stand
  [1,33] = toilet
  [1,34] = sink
  [1,35] = lamp
  [1,36] = bathtub
  [1,37] = bag
}
```

TODO: transalte this.
```python
seg_list = {
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
   14:' desk',
   15:' shelves',
   16:' curtain',
   17:' dresser',
   18:' pillow',
   19:' mirror',
   20:' floor_mat',
   21:'clothes',
   22:' ceiling',
   23:' books',
   24:' fridge',
   25:'tv',
   26:' paper',
   27:' towel',
   28:'shower_curtain',
   29:' box',
   30:' whiteboard',
   31:'person',
   32:' night_stand',
   33:' toilet',
   34:' sink',
   35:' lamp',
   36:' bathtub',
   37:' bag',
}
```

# seg13 list
Label Number	Label Name
1 Bed
2	Books
3	Ceiling
4	Chair
5	Floor
6	Furniture
7	Objects
8	Picture
9	Sofa
10	Table
11	TV
12	Wall
13	Window