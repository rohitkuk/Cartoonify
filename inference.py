
import imp
from statistics import mode
import torch 
from torchvision import transforms
import argparse
from PIL import Image



parser = argparse.ArgumentParser()


parser.add_argument('--img', metavar='img', type=str ,
                    help = "Image to Cartoonize")

args = parser.parse_args()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_DIM      = 512
lr           = 2e-4
MAPS_GEN     = 64
MAPS_DISC    = 64
IMG_CHANNELS = 3
L1_LAMBDA    = 100


Trasforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))
    ])

model = torch.load('weights/Cartoonify_Generator.pt')

model.to(DEVICE)

img = Image.open(args.img)
