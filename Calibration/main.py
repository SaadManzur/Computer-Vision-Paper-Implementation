import cv2 as opencv
import numpy as np
import glob

images = glob.glob('images/*.jpg')

patternX = 8
patternY = 6

criteria = (opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((patternX * patternY, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

points2d = []
points3d = []

for filename in images:
    image = opencv.imread(filename)
    gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)

    ret, corners = opencv.findChessboardCorners(gray, (patternX, patternY), None)

    if ret:
        points3d.append(objp)

        finesedCorners = opencv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        points2d.append(finesedCorners)

        opencv.drawChessboardCorners(image, (patternX, patternY), finesedCorners, ret)

        opencv.imshow(filename, image)
        opencv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = opencv.calibrateCamera(points3d, points2d, gray.shape[::-1], None, None)

print(mtx)
