import numpy as np
import cv2 as opencv


def pca_analysis(points):
    points2d = points.reshape(points.shape[0], points.shape[2]).T

    mean = np.mean(points2d, axis=1)[:, np.newaxis]
    '''
    difference = np.subtract(points2d, mean)

    covariance = difference @ difference.T

    eigen_values, eigen_vectors = np.linalg.eig(covariance)
    '''

    mean, eigen_vectors = opencv.PCACompute(points2d, mean.reshape(1, -1), 2, opencv.PCA_DATA_AS_COL)

    return mean, eigen_vectors

if __name__ == '__main__':
    img = opencv.imread('pca_test1.jpg')
    gray = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)

    ret, binary = opencv.threshold(gray, 50, 255, opencv.THRESH_BINARY | opencv.THRESH_OTSU)

    # opencv.imshow("bin", binary)
    # opencv.waitKey(500)

    image, contours, hierarchy = opencv.findContours(binary, opencv.RETR_LIST, opencv.CHAIN_APPROX_NONE)

    counter = 0
    for contour in contours:
        area = opencv.contourArea(contour)

        if area < 1e2 or area > 1e5:
            continue

        opencv.drawContours(img, contours, counter, color=(0, 0, 255), thickness=2, lineType=opencv.LINE_8)

        center, vecs = pca_analysis(contour)

        opencv.circle(img, (int(center[0, 0]), int(center[1, 0])), 5, color=(255, 0, 0))
        p1 = center + vecs[:, 0] * 10
        p2 = center - vecs[:, 1] * 10
        opencv.line(img, (int(center[0, 0]), int(center[1, 0])),
                    (int(p2[0, 0]), int(p2[1, 0])), color=(0, 255, 0), thickness=2)

        counter = counter + 1

    opencv.imshow('image', img)
    opencv.waitKey(3000)

    opencv.destroyAllWindows()
