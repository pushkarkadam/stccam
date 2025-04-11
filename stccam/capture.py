from harvesters.core import Harvester
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from genicam.gentl import TimeoutException
import sys
import os


def capture_image(genTL_path, resolution=(1920, 1080)):
    """Captures images

    Parameters
    ----------
    genTL_path: str
        Path where the ``.cti`` file from the camera SDK is located.
        For STC camera, this file is located at ``/opt/sentech/lib/<filename>.cti``
        for Linux OS.
    resolution: tuple
        Image resolution as a tuple ``(width, height)``
        This should be more than the mininum resolution and less than the maximum resolution
        the camera can support.

    Returns
    -------
    numpy.ndarray
        A numpy image.

    Examples
    --------
    >>> from stccam import *
    >>> image = capture_image('/path/to/gentl_file.cti', (1920, 1080))
    
    """
    # Extracting the resolution of the image
    width, height = resolution
    
    # Creating harvester object
    h = Harvester()

    # Adding the genTL path
    h.add_file(genTL_path)

    h.update()

    if not len(h.device_info_list):
        print('Camera not found')
        sys.exit(1)

    device_name = h.device_info_list[0].display_name

    print(f'Device: {device_name} detected.')

    ia = h.create(0)

    # Sets the image resolution and the type of image (RGB)
    ia.remote_device.node_map.Width.value = width
    ia.remote_device.node_map.Height.value = height
    ia.remote_device.node_map.PixelFormat.value = 'RGB8'

    ia.start()

    with ia.fetch() as buffer:
        component = buffer.payload.components[0]

        image_data = component.data.reshape(
            component.height, component.width, 3
        )

    return image_data