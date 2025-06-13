import sys

sys.path.append('../')

from stccam import calibration

calibration.stereo_calibration(file_path='../data/calib/13-06-2025-15-20', 
                               chessboard_size=(8,4),
                               square_size=0.03,
                               param_save_path='../data/calib/13-06-2025-15-20',
                               save_rendered='../data/calib/13-06-2025-15-20/rendered'
                              )