from harvesters.core import Harvester
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from genicam.gentl import TimeoutException
import sys
import os
import time


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

def capture_usb_image(genTL_path, 
                      resolution=(1920, 1080), 
                      pixel_format_value='BayerRG8', 
                      image_type=cv2.COLOR_BayerBG2RGB, 
                      save_path=''):
    """Captures images for STC USB camera.

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
    pixel_format_value: str
        Pixel format value for the device.
        This is different for every device.
        A list of available value can be obtained by using the following code:

        .. code-block::python

            from harvesters.core import Harvester
            h = Harvester()
            h.add_file('/path/to/gentl_file.cti')
            h.update()
            ia = h.create(0)
            print(ia.remote_device.node_map.PixelFormat.symbolics)

        A tuple will be printed out as follows:

        .. code-block::

            ('BayerRG8', 'BayerRG10', 'BayerRG10p', 'BayerRG12', 'BayerRG12p')
    image_type: int, default ``cv2.COLOR_BayerBG2RGB``
        This will convert the buffer stream from Bayer to RGB.
        Other option include ``cv2.COLOR_BayerBG2BGR``
    save_path: str, default ``''``
        Path to save the image. Also include the image name.
        When ``save_path`` is empty string, then it will not save the image.
        Example: ``'~/Documents/image.png'``
            
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

    ia.remote_device.node_map.Width.value = width
    ia.remote_device.node_map.Height.value = height
    ia.remote_device.node_map.PixelFormat.value = pixel_format_value

    ia.start()

    with ia.fetch() as buffer:
        component = buffer.payload.components[0]

        image_data = component.data.reshape(
            component.height, component.width
        )

    # Waiting for 0.1 seconds before stopping.
    # This provides enough time for the image acquisition
    time.sleep(0.1)

    # closing stream
    ia.stop()
    ia.destroy()
    h.reset()

    # Converting bayer image to RGB image.
    image = cv2.cvtColor(image_data, image_type)

    if save_path:
        cv2.imwrite(save_path, image)
    
    return image