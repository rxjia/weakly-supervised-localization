import cv2
import numpy as np
import glob
import os
from utils import get_frames_from_a_video

vid_file = '../new_test/20201204_150730.avi'
save_path = '../new_test/images/C'

if not os.path.exists(save_path):
    os.makedirs(save_path)

get_frames_from_a_video(vid_file, save_path)