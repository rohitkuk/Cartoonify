from sklearn import datasets
import torch.nn as nn
from torch.utils.data import Dataset
import os 
from PIL import Image


class CartoonDataset(Dataset):
    def __init__(self, datadir, transforms = None) :
        super(CartoonDataset,self).__init__()
        self.datadir_ = datadir
        self.face_files = [os.path.join(datadir + "/face" , i) for i in os.listdir(datadir+"/face")]
        self.comic_files = [os.path.join(datadir + "/comics" , i) for i in os.listdir(datadir+"/comics")]
        self.transforms = transforms


    def __len__(self):
        return len(self.comic_files)
    

    def __getitem__(self, index):

        face_image = Image.open(self.face_files[index])
        comic_image = Image.open(self.comic_files[index])
        if self.transforms:
            face_image, comic_image = self.transforms(face_image), self.transforms(comic_image)
        return face_image, comic_image


if __name__ == '__main__':
    CartoonDataset()