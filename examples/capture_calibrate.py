import sys 
import matplotlib.pyplot as plt
import cv2

sys.path.append('../')
from stccam import capture 

GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

capture.capture_stereo_calibration(GENTL_PATH,image_type=cv2.COLOR_BayerBG2BGR, save_path='../data/calib')