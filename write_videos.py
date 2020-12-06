import cv2
import numpy as np
import glob
import os
from utils import Drawer

result_name = '2020-12-06_14-44-36'
filepath = f'test_result/{result_name}'
videoname = f'test_result/{result_name}/video_{result_name}.avi'
Drawer.write_video_from_images(filepath, videoname)
print('done')