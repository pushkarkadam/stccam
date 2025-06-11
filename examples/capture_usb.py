import sys 
import matplotlib.pyplot as plt
import cv2

sys.path.append('../')
from stccam import capture 

GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

image = capture.capture_usb_image(GENTL_PATH,resolution=(1280,720), image_type=cv2.COLOR_BayerBG2BGR, save_path='')

print(image.shape)

cv2.imshow('usb camera',image)

cv2.waitKey(0)
cv2.destroyAllWindows()