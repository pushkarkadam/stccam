import sys
import pytest

sys.path.append('../')

from stccam import capture

def test_capture_usb_image():
    """Test for capture_usb_image() function"""
    
    GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

    image = capture.capture_usb_image(genTL_path=GENTL_PATH, resolution=(1920, 1080))

    assert(image.shape == (1080, 1920, 3))

def test_capture_stereo():
    """Test for capture_stereo() function"""

    GENTL_PATH = '/opt/sentech/lib/libstgentl.cti'

    images = capture.capture_stereo(genTL_path=GENTL_PATH, resolution=(1920, 1080))

    imageL, imageR = images

    assert(type(images) == list)
    assert(imageL.shape == (1080, 1920, 3))
    assert(imageR.shape == (1080, 1920, 3))