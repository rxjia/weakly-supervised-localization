import cv2
import numpy as np
import glob
import os
from utils import Drawer

root = f'/home/yanglei/codes/WSOL/videos/'
vid_name = 'VID_20201207_150008.mp4'

vid_file = os.path.join(root, vid_name)
save_path = os.path.join(root, vid_name).split('.')[0]

print(vid_file)
print(save_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)

Drawer.get_frames_from_a_video(vid_file, save_path)