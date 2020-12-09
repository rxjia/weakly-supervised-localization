import cv2
import numpy as np
import glob
import os
from utils import Drawer

# test_path = 'test_result'
test_path = 'detection'
result_name = '2020-12-08_18-06-40'
filepath = f'{test_path}/{result_name}'
videoname = f'{test_path}/{result_name}/video_{result_name}.avi'
Drawer.write_video_from_images(filepath, videoname, deleteImg=True)
print('done')