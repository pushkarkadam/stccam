from harvesters.core import Harvester
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from genicam.gentl import TimeoutException
import sys
import os
import time


def capture_poe_image(genTL_path, resolution=(1920, 1080)):
    """Captures images.
    This function is developed to work with PoE camera.

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
    >>> from stccam import capture
    >>> image = capture.capture_poe_image('/path/to/gentl_file.cti', (1920, 1080))
    
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
    r"""Captures images for STC USB camera.

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
        Options include: ``('BayerRG8', 'BayerRG10', 'BayerRG10p', 'BayerRG12', 'BayerRG12p')``
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
    >>> from stccam import capture
    >>> image = capture.capture_usb_image('/path/to/gentl_file.cti', (1920, 1080))
    
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
    time.sleep(1)

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

def capture_stereo(genTL_path, 
                   cam_serial = {'left': '24MB632', 'right': '24MB633'},
                   resolution=(1920, 1080),
                   pixel_format_value='BayerRG8',
                   image_type=cv2.COLOR_BayerBG2RGB,
                   save_path=''
                  ):
    """Captures stereo image.
    This function can be only be used with STC USB camera.
    
    Parameters
    ----------
    genTL_path: str
        Path where the ``.cti`` file from the camera SDK is located.
        For STC camera, this file is located at ``/opt/sentech/lib/<filename>.cti``
        for Linux OS.
    cam_serial: dict, default ``{'left': '24MB632', 'right': '24MB633'}``
        A dictionary of serial number for the camera that maps the left and right of the stereo
        to the serial number.
    resolution: tuple
        Image resolution as a tuple ``(width, height)``
        This should be more than the mininum resolution and less than the maximum resolution
        the camera can support.
    pixel_format_value: str
        Pixel format value for the device.
        This is different for every device.
        Options: ``('BayerRG8', 'BayerRG10', 'BayerRG10p', 'BayerRG12', 'BayerRG12p')``
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
    >>> from stccam import capture
    >>> image = capture.capture_stereo('/path/to/gentl_file.cti', (1920, 1080))
    
    """
    # Extracting the resolution of the image
    width, height = resolution
    
    # Creating harvester object
    h = Harvester()

    # Adding the genTL path
    h.add_file(genTL_path)

    h.update()

    devices = h.device_info_list

    if not len(devices):
        print('Camera not found')
        sys.exit(1)

    print("Following devices detected")
    for device in devices:
        print(device.display_name)

    # Checks if two cameras are detected
    try:
        assert(len(devices) == 2)
    except Exception as e:
        print(e)
        print(f"Cameras found {len(devices)} != 2")


    # Storing stereo images left, right
    stereo_images = []

    # Creating image acquirer for both left and right camera
    ias = [h.create(search_key={'serial_number': v}) for v in cam_serial.values()]

    for ia in ias:
        ia.remote_device.node_map.Width.value = width
        ia.remote_device.node_map.Height.value = height
        ia.remote_device.node_map.PixelFormat.value = pixel_format_value
    
        ia.start()

    # creating two separe image acquirer for left and right camera
    ia_left, ia_right = ias

    with ia_left.fetch() as buffer_left, ia_right.fetch() as buffer_right:
        # left image
        component_left = buffer_left.payload.components[0]
        image_data_left = component_left.data.reshape(
            component_left.height, component_left.width
        )
        
        image_left = cv2.cvtColor(image_data_left, image_type)

        stereo_images.append(image_left)

        # right image
        component_right = buffer_right.payload.components[0]
        image_data_right = component_right.data.reshape(
            component_right.height, component_right.width
        )
        
        image_right = cv2.cvtColor(image_data_right, image_type)

        stereo_images.append(image_right)
    
    for ia in ias:
        ia.stop()
        ia.destroy()
    
    h.reset()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        left_save_path = os.path.join(save_path, 'left.png')
        right_save_path = os.path.join(save_path, 'right.png')
        cv2.imwrite(left_save_path, stereo_images[0])
        cv2.imwrite(right_save_path, stereo_images[1])

    return stereo_images

def capture_stereo_calibration(genTL_path, 
                               cam_serial = {'left': '24MB632', 'right': '24MB633'},
                               resolution=(1920, 1080),
                               pixel_format_value='BayerRG8',
                               image_type=cv2.COLOR_BayerBG2RGB,
                               save_path=''):
    """Captures images for stereo calibration.

    Parameters
    ----------
    genTL_path: str
        Path where the ``.cti`` file from the camera SDK is located.
        For STC camera, this file is located at ``/opt/sentech/lib/<filename>.cti``
        for Linux OS.
    cam_serial: dict, default ``{'left': '24MB632', 'right': '24MB633'}``
        A dictionary of serial number for the camera that maps the left and right of the stereo
        to the serial number.
    resolution: tuple
        Image resolution as a tuple ``(width, height)``
        This should be more than the mininum resolution and less than the maximum resolution
        the camera can support.
    pixel_format_value: str
        Pixel format value for the device.
        This is different for every device.
        Options: ``('BayerRG8', 'BayerRG10', 'BayerRG10p', 'BayerRG12', 'BayerRG12p')``
    image_type: int, default ``cv2.COLOR_BayerBG2RGB``
        This will convert the buffer stream from Bayer to RGB.
        Other option include ``cv2.COLOR_BayerBG2BGR``
    save_path: str, default ``''``
        Path to save the image. Also include the image name.
        When ``save_path`` is empty string, then it will not save the image.
        This will create two directories: ``save_path/<date time>/stereo_left`` and ``save_path/<date time>/stereo_right``
            
    Returns
    -------
    numpy.ndarray
        A numpy image.

    Examples
    --------
    >>> from stccam import capture
    >>> capture.capture_stereo_calibration('/path/to/gentl_file.cti', (1920, 1080))
    
    """

    # Extracting the resolution of the image
    width, height = resolution
    
    # Creating harvester object
    h = Harvester()
    
    # Adding the genTL path
    h.add_file(genTL_path)
    
    h.update()
    
    devices = h.device_info_list
    
    if not len(devices):
        print('Camera not found')
        sys.exit(1)
    
    print("Following devices detected")
    for device in devices:
        print(device.display_name)
    
    # Checks if two cameras are detected
    try:
        assert(len(devices) == 2)
    except Exception as e:
        print(e)
        print(f"Cameras found {len(devices)} != 2")
    
    # date
    ct = datetime.datetime.now()
    date = ct.strftime("%d-%m-%Y-%H-%M")
    
    # Make directory
    path_left = os.path.join(save_path, date, 'stereo_left')
    path_right = os.path.join(save_path, date, 'stereo_right')
    
    print(f"Creating directories: \n{path_left}\n{path_right}")
    os.makedirs(path_left, exist_ok=True)
    os.makedirs(path_right, exist_ok=True)
    
    # Creating image acquirer for both left and right camera
    ias = [h.create(search_key={'serial_number': v}) for v in cam_serial.values()]
    
    for ia in ias:
        ia.remote_device.node_map.Width.value = width
        ia.remote_device.node_map.Height.value = height
        ia.remote_device.node_map.PixelFormat.value = pixel_format_value
    
        ia.start()
    
    # creating two separe image acquirer for left and right camera
    ia_left, ia_right = ias
    
    num_images = 0
    
    try:
        while True:
            with ia_left.fetch() as buffer_left, ia_right.fetch() as buffer_right:
                # left image
                component_left = buffer_left.payload.components[0]
                image_data_left = component_left.data.reshape(
                    component_left.height, component_left.width
                )
                
                image_left = cv2.cvtColor(image_data_left, image_type)
        
                # right image
                component_right = buffer_right.payload.components[0]
                image_data_right = component_right.data.reshape(
                    component_right.height, component_right.width
                )
                
                image_right = cv2.cvtColor(image_data_right, image_type)
        
                cv2.imshow('left_camera', image_left)
                cv2.imshow('right_camera', image_right)
        
                key = cv2.waitKey(5)
        
                # Check if ESC key is pressed whose ASCII value is 27
                if key == 27:
                    break
                elif key == ord('s'):
                    cv2.imwrite(os.path.join(path_left, 'imageL_' + str(num_images) + '.png'), image_left)
                    cv2.imwrite(os.path.join(path_right, 'imageR_' + str(num_images) + '.png') , image_right)
                    print('\033[92m' + "Images saved!")
                    num_images += 1
    except KeyboardInterrupt:
        print("Interupted by user.")
    
    finally:
        for ia in ias:
            ia.stop()
            ia.destroy()
        h.reset()
        cv2.destroyAllWindows()

def live_stream_stereo(genTL_path, 
                       cam_serial = {'left': '24MB632', 'right': '24MB633'},
                       resolution=(1920, 1080),
                       pixel_format_value='BayerRG8',
                       image_type=cv2.COLOR_BayerBG2RGB,
                       show_combined=True):
    """Live streaming from stereo setup.
    
    Parameters
    ----------
    genTL_path: str
        Path where the ``.cti`` file from the camera SDK is located.
        For STC camera, this file is located at ``/opt/sentech/lib/<filename>.cti``
        for Linux OS.
    cam_serial: dict, default ``{'left': '24MB632', 'right': '24MB633'}``
        A dictionary of serial number for the camera that maps the left and right of the stereo
        to the serial number.
    resolution: tuple
        Image resolution as a tuple ``(width, height)``
        This should be more than the mininum resolution and less than the maximum resolution
        the camera can support.
    pixel_format_value: str
        Pixel format value for the device.
        This is different for every device.
        Options: ``('BayerRG8', 'BayerRG10', 'BayerRG10p', 'BayerRG12', 'BayerRG12p')``
    image_type: int, default ``cv2.COLOR_BayerBG2RGB``
        This will convert the buffer stream from Bayer to RGB.
        Other option include ``cv2.COLOR_BayerBG2BGR``
    show_combined: bool, default ``True``
        Shows the stereo images combined.
        Due to the large size of the image input, the image is resized to ``(640, 480)``
        resolution for each images from the stereo setup.

    Examples
    --------
    >>> from stccam import capture
    >>> capture.live_stream_stereo(genTL_path='/opt/sentech/lib/libstgentl.cti', resolution=(1280, 720), image_type=cv2.COLOR_BayerBG2BGR, show_combined=False)
    
    """

    # Extracting the resolution of the image
    width, height = resolution
    
    # Creating harvester object
    h = Harvester()
    
    # Adding the genTL path
    h.add_file(genTL_path)
    
    h.update()
    
    devices = h.device_info_list
    
    if not len(devices):
        print('Camera not found')
        sys.exit(1)
    
    print("Following devices detected")
    for device in devices:
        print(device.display_name)
    
    # Checks if two cameras are detected
    try:
        assert(len(devices) == 2)
    except Exception as e:
        print(e)
        print(f"Cameras found {len(devices)} != 2")
    
    # Creating image acquirer for both left and right camera
    ias = [h.create(search_key={'serial_number': v}) for v in cam_serial.values()]
    
    for ia in ias:
        ia.remote_device.node_map.Width.value = width
        ia.remote_device.node_map.Height.value = height
        ia.remote_device.node_map.PixelFormat.value = pixel_format_value
    
        ia.start()
    
    # creating two separe image acquirer for left and right camera
    ia_left, ia_right = ias

    print("Starting live stream. Press ESC to stop")
    
    try:
        while True:
            with ia_left.fetch() as buffer_left, ia_right.fetch() as buffer_right:
                # left image
                component_left = buffer_left.payload.components[0]
                image_data_left = component_left.data.reshape(
                    component_left.height, component_left.width
                )
                
                image_left = cv2.cvtColor(image_data_left, image_type)
        
                # right image
                component_right = buffer_right.payload.components[0]
                image_data_right = component_right.data.reshape(
                    component_right.height, component_right.width
                )
                
                image_right = cv2.cvtColor(image_data_right, image_type)

                if show_combined:
                    imgL = cv2.resize(image_left, (640, 480))
                    imgR = cv2.resize(image_right, (640, 480))
                    stereo_image = np.hstack((imgL, imgR))
                    cv2.imshow("stereo", stereo_image)

                else:
                    cv2.imshow('left_camera', image_left)
                    cv2.imshow('right_camera', image_right)
        
                key = cv2.waitKey(5)
        
                # Check if ESC key is pressed whose ASCII value is 27
                if key == 27:
                    break

    except KeyboardInterrupt:
        print("Interupted by user.")
    
    finally:
        for ia in ias:
            ia.stop()
            ia.destroy()
        h.reset()
        cv2.destroyAllWindows()