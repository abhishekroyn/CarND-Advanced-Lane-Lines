import numpy as np
import cv2
import glob
import pickle
from tracker import Tracker

def find_camera_calibration(images, cal_width = 9, cal_height = 6):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) , ... , (6,5,0)
    objp = np.zeros((cal_width * cal_height,3), np.float32)
    objp[:,:2] = np.mgrid[0:cal_width, 0:cal_height].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cal_width, cal_height), None)

        # If found, add object points, image points
        if ret == True:
            print('working on ', fname)
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cal_width, cal_height), corners, ret)
            write_name = '../camera_cal/corners_found'+str(idx+1)+'.jpg'
            cv2.imwrite(write_name, img)

    # load image for reference
    img = cv2.imread('../camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("../camera_cal/calibration_pickle.p", "wb"))

# Set up the overall class to do all the tracking
tracker = Tracker(Mycalib_width = 9, Mycalib_height = 6, Mywindow_width = 25, Mywindow_height = 80, Mymargin = 50, Myminpix = 50000, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15, Mybot_width = .76, Mymid_width = .08, Myheight_pct = .62, Mybottom_trim = .935)

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# calibrate camera
find_camera_calibration(images, tracker.cal_width, tracker.cal_height)
