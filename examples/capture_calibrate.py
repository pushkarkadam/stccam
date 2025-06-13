import sys 
import matplotlib.pyplot as plt
import cv2

sys.path.append('../')
from stccam import capture 

GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

capture.capture_stereo_calibration(GENTL_PATH, save_path='../data/calib')