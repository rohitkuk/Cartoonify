import os 
from glob import glob 

dirs_ = ['assets', 'data', 'docs','logs', 'pipelines', 'research', 'src','tests', 'weights']


# Make it more sesnible use dir or file exists.
for dir_ in dirs_:
    try:
        os.mkdir(dir_)
        open(dir_ + '/__init__.py', 'w')
        print("Directory Named {} created Successfully".format(dir_))
        print("File at {} created Successfully".format(dir_ + '/__init__.py'))
    except:
        pass

files_ = ["requirements.txt", '__init__.py']


for file_ in files_:
    try:
        open(file_, 'w')
        print("File named {} created Successfully".format(file_))
    except:
        pass