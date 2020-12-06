import cv2
import numpy as np
import glob
import os
from utils import Drawer

vid_file = '/home/yanglei/codes/WSOL/20201204_150730_convoybelt.avi'
save_path = '../new_test_no_resize/images/C'

if not os.path.exists(save_path):
    os.makedirs(save_path)

Drawer.get_frames_from_a_video(vid_file, save_path)