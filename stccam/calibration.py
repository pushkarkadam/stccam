import numpy as np 
import cv2
import glob
import os
from tqdm import tqdm


def stereo_calibration(file_path, 
                       chessboard_size, 
                       square_size=0.03,
                       chessboard_flag=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                       cornerSubPix_winSize=(11,11),
                       cornerSubPix_zeroZone=(-1,-1),
                       cornerSubPix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                       calibrateCamera_flags=cv2.CALIB_ZERO_TANGENT_DIST,
                       stereoCalibrate_flags=cv2.CALIB_FIX_INTRINSIC,
                       param_save_path='.',
                       image_limit=10,
                       save_rendered=None
                      ):
    """Calibrates stereo camera.

    For high quality result, use at least 10 images of a ``7 x 8`` or larger chessboard.
    
    Parameters
    ----------
    file_path: str
        Path of the file.
    chessboard_size: tuple
        Size of the grid of the chessboard.
        For a chess board of ``9 x 8`` pattern, use the input as ``(8, 7)``.
    square_size: float, default ``0.03``.
        The size of the square of the chessboard. 
        The default dimension is in meters.
        If other units are chosen, make sure to stay consistent with the units in depth detection as well.
    chessboard_flag: int, default ``cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE``
        Uses the threshold from the enum provided in opencv.
    cornerSubPix_winSize: tuple, default ``(11,11)``
        A tuple of ``int`` that uses the kerner size refined corners.
    cornerSubPix_zeroZone: tuple, default ``(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)``
        A tuple of critereo that is used in fine grained sub pixed calculations.
    calibrateCamera_flags: int, default ``cv2.CALIB_ZERO_TANGENT_DIST``
        Calibrate camera flags
    stereoCalibrate_flags: int, default ``cv2.CALIB_FIX_INTRINSIC``
        Flags for calibrating stereo camera.
    param_save_path: str, default ``'.'``
        Path to save the calibration parameters.
    image_limit: int, default ``10``
        Limit the images. 
        It turns out that more image increases the reprojection error.
    save_rendered: str, None
        Path to save the rendered images.

    Returns
    -------
    retval: float
        Reprojection error.
        A value less than ``1`` is acceptable. 
        A good result would be to have half a pixel i.e. ``0.5`` error.

    Examples
    --------
    >>> from stccam import calibration
    >>> calibration.stereo_calibration(file_path='../data/calib/13-06-2025-15-20', chessboard_size=(8,4), square_size=0.03, param_save_path='../data/calib/13-06-2025-15-20', save_rendered='../data/calib/13-06-2025-15-20/rendered')
    
    """

    # Organising the stereo image paths.
    calib_path = {x: os.path.join(file_path, f"stereo_{x}") for x in ["left", "right"]}

    # Load stereo image pairs
    # Unrapping the paths in tuple
    left_images_path, right_images_path = tuple(calib_path.values())


    # extracting the chessboard size
    ch_r, ch_c = chessboard_size

    # Creating object points
    objp = np.zeros((ch_r * ch_c, 3), np.float32)

    # reshaping to match t
    objp[:, :2] = np.mgrid[0:ch_r, 0:ch_c].T.reshape(-1, 2)

    objp *= square_size

    # list to store object points and image points from both cameras
    # 3D real world space of the chess board
    # ``objpoints`` is the list of the list of ``objp`` for each set of stereo images  
    objpoints = []

    # 2D image points in the left and right image frame
    # Consists of the list of list of image points per set of stereo images
    imgpoints_left = []
    imgpoints_right = []

    # Creating a list of the images from the path
    left_images = sorted(glob.glob(os.path.join(left_images_path, "*.png")))[:image_limit]
    right_images = sorted(glob.glob(os.path.join(right_images_path, "*.png")))[:image_limit]

    # Detect chessboard corners
    for idx, (left_img, right_img) in enumerate(zip(left_images, right_images)):
        # Reading the images
        imgL = cv2.imread(left_img)
        imgR = cv2.imread(right_img)

        # Convering the images from BGR to Gray
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Chessboard corners for left and right images
        retL, cornersL = cv2.findChessboardCorners(image=grayL, 
                                                   patternSize=chessboard_size, 
                                                   flags=chessboard_flag)
        
        retR, cornersR = cv2.findChessboardCorners(image=grayR, 
                                                   patternSize=chessboard_size, 
                                                   flags=chessboard_flag)

        if retL and retR:
            # adding the object points of the current set of images to the list of objpoints for all the images
            objpoints.append(objp)

            # Calculating subpixel to get more accurate result
            cornersL = cv2.cornerSubPix(image=grayL, corners=cornersL, winSize=cornerSubPix_winSize, zeroZone=cornerSubPix_zeroZone, criteria=cornerSubPix_criteria)

            cornersR = cv2.cornerSubPix(image=grayR, corners=cornersR, winSize=cornerSubPix_winSize, zeroZone=cornerSubPix_zeroZone, criteria=cornerSubPix_criteria)

            # Appending list of refined cornerpoints to the imagepoints
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            
            # performs rendering and saves the imamge to save_rendered directory
            if save_rendered:
                left_path = os.path.join(save_rendered, 'stereo_left')
                right_path = os.path.join(save_rendered, 'stereo_right')
                # Creates directory if it does not exist
                if not os.path.exists(save_rendered):
                    # os.makedirs(save_rendered)
                    
                    os.makedirs(left_path)
                    os.makedirs(right_path)
                    
                
                cv2.drawChessboardCorners(image=imgL, patternSize=chessboard_size, corners=cornersL, patternWasFound=retL)
                cv2.drawChessboardCorners(image=imgR, patternSize=chessboard_size, corners=cornersR, patternWasFound=retR)

                cv2.imwrite(os.path.join(left_path, f"left_img{idx}.png"), imgL)
                cv2.imwrite(os.path.join(right_path, f"right_img{idx}.png"), imgR)

    # extracting shape of the image from last grayL in the list
    img_size=grayL.shape[::-1]

    # Calibrating individual camera matrix and distortion coefficients to use later in stereoCalibrate for more precision
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints_left, imageSize=img_size, cameraMatrix=None, distCoeffs=None, flags=calibrateCamera_flags)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints_right, imageSize=img_size, cameraMatrix=None, distCoeffs=None, flags=calibrateCamera_flags)

    # Calibrate stereo
    retval, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(objectPoints=objpoints,
                                                                       imagePoints1=imgpoints_left,
                                                                       imagePoints2=imgpoints_right,
                                                                       cameraMatrix1=mtxL,
                                                                       distCoeffs1=distL,
                                                                       cameraMatrix2=mtxR,
                                                                       distCoeffs2=distR,
                                                                       imageSize=img_size,
                                                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                                                                       flags=stereoCalibrate_flags)

    print(f"Calibration RMS error: {retval}")

    # Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=mtxL, distCoeffs1=distL, cameraMatrix2=mtxR, distCoeffs2=distR, imageSize=img_size, R=R, T=T)
    # save calibration results
    save_path = os.path.join(param_save_path, 'stereo_calib.npz')
    
    np.savez(save_path, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
        
    return retval