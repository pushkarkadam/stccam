import sys 
import matplotlib.pyplot as plt
import cv2

sys.path.append('../')
from stccam import capture 

GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

# Declare the cameras of stereo system by assigning the camera serial number.
camera_serial = {'left': '24MB632', 'right': '24MB633'}

left_image, right_image = capture.capture_stereo(GENTL_PATH, cam_serial=camera_serial ,resolution=(1280,720), image_type=cv2.COLOR_BayerBG2BGR, save_path='')

print(left_image.shape)
print(right_image.shape)

cv2.imshow('left_image',left_image)
cv2.imshow('right_image', right_image)

cv2.waitKey(0)
cv2.destroyAllWindows()