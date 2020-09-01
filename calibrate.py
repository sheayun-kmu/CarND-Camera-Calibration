import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[1::-1], None, None
    )
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# prepare object points
nx = 8 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

objpoints = []
imgpoints = []

objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Make a list of calibration images
images = glob.glob('./calibration_wide/GOPR????.jpg')
for fname in images:
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points & image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
    else:
        print("Failed to find %d * %d corners from image %s" % (nx, ny, fname))

img = cv2.imread('./calibration_wide/test_image.jpg')
dst = cal_undistort(img, objpoints, imgpoints)
plt.imshow(dst)
plt.show()
