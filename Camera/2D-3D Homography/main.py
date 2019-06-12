"""
Projection matrix recovery using planar homography between model and image plane using Zhang's
method.

Python: 3.6
OpenCV: 3.4.2
"""

from matplotlib import pyplot as plt
import cv2 as opencv
import numpy as np

MARKER_SIZE = 19.6

# Calibration Matrix obtained from calibration part under the camera section
# MI 6X
'''
K = np.array([[3.39957900e+03, 0.00000000e+00, 1.98170992e+03],
              [0.00000000e+00, 3.40113884e+03, 1.46044654e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
'''

# Logitech C920
K = np.array([[662.25801378, 0., 294.92509514],
              [0., 663.05010424, 209.48385376],
              [0., 0., 1.]])


def locate_marker(marker_id, image_name, show=False):
    input_img = opencv.imread(image_name)
    # input_img = opencv.resize(input_img, (int(input_img.shape[1] / 5), int(input_img.shape[0] / 5)))

    marker_dictionary = opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_7X7_1000)

    corners, ids, rejected_img_points = opencv.aruco.detectMarkers(input_img, marker_dictionary)

    points3d = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2],
                         [MARKER_SIZE / 2, MARKER_SIZE / 2],
                         [MARKER_SIZE / 2, -MARKER_SIZE / 2],
                         [-MARKER_SIZE / 2, -MARKER_SIZE / 2]])

    if show:
        output = opencv.aruco.drawDetectedMarkers(input_img, corners, ids)

        for i in range(ids.shape[0]):
            if ids[i] == marker_id:
                counter = 0
                for corner in corners[i][0]:
                    coord = '(' + str(points3d[counter, 0]) + ', ' + str(points3d[counter, 1]) + ', 0)'
                    opencv.circle(output, (corner[0], corner[1]), 4, (255, 0, 0), -1)
                    opencv.putText(output, coord,
                                   (int(corner[0] - 20),
                                    int(corner[1] - (-1) ** (counter // 2) * 20)),
                                   opencv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))
                    counter = counter + 1

        opencv.imshow('marker', output)
        opencv.waitKey(10000)

    return corners[0][0], points3d


def get_extrinsic(img_points, points_3d):
    img_points_h = np.hstack((img_points, np.ones((4, 1))))
    normalized_img_points = np.linalg.inv(K) @ img_points_h.T

    H, mask = opencv.findHomography(points_3d, normalized_img_points.T[:, :2])
    H /= np.linalg.norm(H[:, 0])

    r1 = H[:, 0]
    r2 = H[:, 1]
    r3 = np.cross(r1, r2)

    t = (2 * H[:, 2]) / (np.linalg.norm(r1) + np.linalg.norm(r2))

    R = np.hstack((r1[:, np.newaxis], r2[:, np.newaxis], r3[:, np.newaxis]))

    U, E, Vt = np.linalg.svd(R)

    R = U @ Vt

    return R, t[:, np.newaxis]


def get_projection_matrix(rotation, translation):
    extrinsic = np.hstack((rotation, translation))
    return K @ extrinsic


def show_axis(image_name, P, scale=0.2, thickness=20):
    input_img = opencv.imread(image_name)

    axis_3d = np.vstack((np.identity(3) * 15, np.ones((1, 3))))
    axis_2d = P @ axis_3d

    origin_3d = np.vstack((np.zeros((3, 1)), np.ones((1, 1))))
    origin_2d = P @ origin_3d
    origin_2d /= origin_2d[2, 0]

    # opencv.circle(input_img, (int(origin_2d[0, 0]), int(origin_2d[1, 0])), 20, (128, 135, 12), -1)

    for i in range(3):
        color = np.zeros((3,))
        color[2 - i] = 255
        axis_2d[:, i] /= axis_2d[2, i]
        opencv.arrowedLine(input_img,
                           (int(origin_2d[0, 0]), int(origin_2d[1, 0])),
                           (int(axis_2d[0, i]), int(axis_2d[1, i])),
                           (color[0], color[1], color[2]),
                           thickness
                           )

    opencv.imwrite('axis_image.jpg', input_img)

    resized_img = opencv.resize(input_img, (int(input_img.shape[1] * scale), int(input_img.shape[0] * scale)))
    opencv.imshow("origin", resized_img)
    opencv.waitKey(5000)


if __name__ == '__main__':
    filename = 'resources/logitech c920/engg_m_test6.jpg'

    img_points, points_3d = locate_marker(20, filename)

    R, t = get_extrinsic(img_points, points_3d)

    P = get_projection_matrix(R, t)
    print(P)

    show_axis(filename, P, scale=1, thickness=2)
