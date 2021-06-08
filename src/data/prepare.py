from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from tqdm import tqdm
import os 
import shutil
from glob import glob
import matplotlib.pyplot as plt



def Download_Dataset(out_path):
    # Initiating the Kaggle API
    api = KaggleApi()
    # Authenticating the Kaggle API
    api.authenticate()

    # Downloading the dataser
    api.dataset_download_files('defileroff/comic-faces-paired-synthetic', quiet=False)

    file_name = 'comic-faces-paired-synthetic.zip'

    with ZipFile(file=file_name) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path= out_path)

    print("#######Deleting Zip File #####")
    os.remove(file_name)

    shutil.move(out_path+ '/face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/comics', out_path+ '/comics')
    shutil.move(out_path+ '/face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/face', out_path+ '/face')
    shutil.rmtree(out_path+ '/samples')
    shutil.rmtree(out_path+ '/face2comics_v1.0.0_by_Sxela')

    comicfiles = glob(out_path+ '/comics/*')
    facefiles = glob(out_path+ '/face/*')

    print("Comics Images Count :: {}".format(len(comicfiles)))
    print("Face Images Count :: {}".format(len(facefiles)))