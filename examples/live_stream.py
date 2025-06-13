import sys 
import matplotlib.pyplot as plt
import cv2

sys.path.append('../')
from stccam import capture 

GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

capture.live_stream_stereo(genTL_path=GENTL_PATH, resolution=(1280, 720), image_type=cv2.COLOR_BayerBG2BGR, show_combined=True)