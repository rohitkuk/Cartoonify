
# tree --dirsfirst --noreport -I 'Dataset*|wandb*|__pycache__|__init__.py|logs|SampleImages|List.md' > List.md 
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn 
from torch import optim


from src.data import dataset
from src.data import prepare
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.utils import utils


import sys
import wandb
import argparse
import shutil
import os
from tqdm import tqdm 
from IPython import get_ipython

import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter


os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser(description='######DImage-to-Image Translation with Conditional Adversarial Nets########')


parser.add_argument('--wandbkey', metavar='wandbkey', default=None,
                    help='Key for Weight and Biases Integration')


parser.add_argument('--projectname', metavar='projectname', default="Cartoonify",
                    help='Key for Weight and Biases Integration')


parser.add_argument('--wandbentity', metavar='wandbentity',
                    help='Entity for Weight and Biases Integration')


parser.add_argument('--tensorboard', metavar='tensorboard', type=bool, default=True,
                    help='Tensorboard Integration')

parser.add_argument('--kaggle_user',default = None,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")


parser.add_argument('--kaggle_key', default = None,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")


parser.add_argument('--batch_size', metavar='batch_size', type=int , default = 32,
                    help = "Batch_Size")
                    

parser.add_argument('--epoch', metavar='epoch', type=int ,default = 5,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")


parser.add_argument('--load_checkpoints', metavar='load_checkpoints',    default = False,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")
                    
args = parser.parse_args()



shutil.rmtree("logs") if os.path.isdir("logs") else ""


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS   = args.epoch
IMG_DIM      = 512
lr           = 2e-4
BATCH_SIZE   = args.batch_size
MAPS_GEN     = 64
MAPS_DISC    = 64
IMG_CHANNELS = 3
L1_LAMBDA    = 100

GEN_CHECKPOINT = '{}_Generator.pt'.format(args.projectname)
DISC_CHECKPOINT = '{}Discriminator.pt'.format(args.projectname)


# Downloading the dataset 

prepare.Download_Dataset(out_path='data')

# Transforms
Trasforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))
    ])


# Data Loaders
train_dataset = dataset.CartoonDataset(datadir='data', transforms = Trasforms)

# Add Num workers
train_loader   = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)


if args.wandbkey :
    wandb_integration = True
    wandb.login(key = args.wandbkey)
    wandb.init(project = args.projectname,  entity=args.wandbentity, resume=True)
    print(wandb.run.name)


# Loading Generator
if os.path.isdir(os.path.join(wandb.run.dir, GEN_CHECKPOINT)) and args.load_checkpoints:
    generator = torch.load(wandb.restore(GEN_CHECKPOINT).name)
else:
    generator = Generator(img_channels=IMG_CHANNELS, features=MAPS_GEN).to(DEVICE)


# Loading Discriminator
if os.path.isdir(os.path.join(wandb.run.dir, DISC_CHECKPOINT)) and args.load_checkpoints:
    discriminator = torch.load(wandb.restore(DISC_CHECKPOINT).name)
else:
    discriminator = Discriminator(img_channels = IMG_CHANNELS, features = MAPS_DISC).to(DEVICE)


# weights Initialize
utils.initialize_weights(generator)
utils.initialize_weights(discriminator)


# Loss and Optimizers
gen_optim = optim.Adam(params = generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(params = discriminator.parameters(), lr=lr, betas=(0.5, 0.999))



BCE = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


# Tensorboard Implementation
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")



if args.wandbkey :
    wandb.watch(generator)
    wandb.watch(discriminator)


# Code for COLLAB TENSORBOARD VIEW
try:
    get_ipython().magic("%load_ext tensorboard")
    get_ipython().magic("%tensorboard --logdir logs")
except:
    pass


# training
discriminator.train()
generator.train()
step = 0
images = []


for epoch in range(1, NUM_EPOCHS+1):
    tqdm_iter = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)

    for batch_idx, (face_image,comic_image) in tqdm_iter:

        face_image,comic_image= face_image.to(DEVICE) , comic_image.to(DEVICE)
        
        # Train Discriminator
        fake_image = generator(face_image)
        
        disc_real = discriminator(face_image, comic_image)
        disc_fake = discriminator(face_image, fake_image)

        disc_real_loss = BCE(disc_real, torch.ones_like(disc_real))
        disc_fake_loss = BCE(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_real_loss+disc_fake_loss)/2

        discriminator.zero_grad()
        disc_loss.backward()
        disc_optim.step()



    # training Generator
        fake_image      = generator(face_image) # WOuld make more fast if we remove this but want to make it more expressive
        disc_fake       = discriminator(face_image, fake_image)
        gen_fake_loss   = BCE(disc_fake, torch.ones_like(disc_fake))
        L1              = l1_loss(fake_image, comic_image) * L1_LAMBDA

        gen_loss        = gen_fake_loss+ L1
        
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        tqdm_iter.set_postfix(
            D_real=torch.sigmoid(disc_real).mean().item(),
            D_fake=torch.sigmoid(disc_fake).mean().item(),
            disc_loss = disc_loss.item(),
            gen_loss  = gen_loss.item()
        )

        if batch_idx % 100 == 0:
            torch.save(generator.state_dict(), os.path.join("weights", GEN_CHECKPOINT))
            torch.save(discriminator.state_dict(), os.path.join("weights", DISC_CHECKPOINT))

            fake_image = fake_image * 0.5 + 0.5 
            face_image = face_image * 0.5 + 0.5 
            
            if args.tensorboard:
                img_grid_real = make_grid(face_image[:8], normalize=True)
                img_grid_fake = make_grid(fake_image[:8], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step +=1
                images.append(img_grid_fake.cpu().detach().numpy())


            if args.wandbkey:
                wandb.log({"Discriminator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
                wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})

                torch.save(generator.state_dict(), os.path.join(wandb.run.dir, GEN_CHECKPOINT))
                torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, DISC_CHECKPOINT))



try:
    matplotlib.rcParams['animation.embed_limit'] = 2**64
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = []
    for j,i in tqdm(enumerate(images)):
        ims.append([plt.imshow(np.transpose(i,(1,2,0)), animated=True)]) 
        
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    f = "animation{}.gif".format(datetime.datetime.now()).replace(":","")

    ani.save(os.path.join(wandb.run.dir,f), writer=PillowWriter(fps=20)) 
    ani.save(f, writer=PillowWriter(fps=20)) 
except:
    pass
    