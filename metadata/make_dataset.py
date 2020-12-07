import numpy as np
import os,sys,inspect
import glob

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import *



ALLOWED_EXTENSIONS = ('.jpg')

def make_dataset(root):
    root = os.path.expanduser(root)

    ## helper functions
    def has_file_allowed_extension(filename, extensions):
        return filename.lower().endswith(extensions)
        
    def is_valid_file(x):
        return has_file_allowed_extension(x, ALLOWED_EXTENSIONS)

    image_list = []
    for root, dirs, files in os.walk(root, topdown=False):
        for file_name in files:
            if not is_valid_file(file_name):
                continue
            file_name_w_root = os.path.join(root, file_name)
            print("files: ", file_name_w_root)
            image_list.append(file_name_w_root)
        # for dir_name in dirs:
        #     print("dirs: ", os.path.join(root, dir_name))
        #     dir_list.append(dir_name)
    
    # sort
    image_list.sort()
    return image_list


if __name__ == '__main__':
    root = '/home/yanglei/codes/WSOL/detection'
    phase = 'detection'
    path = os.path.join(root, phase)
    image_list = make_dataset(path)
    
    write_json_no_indent(image_list, f'{phase}_images.json')

    
