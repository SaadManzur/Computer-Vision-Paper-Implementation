"""
Tested on Python 3.6 and OpenCV 3.4.2
"""

import cv2 as opencv

marker_dictionary = opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_7X7_1000)


def draw_marker(marker_id):
    img = opencv.aruco.drawMarker(marker_dictionary, marker_id, 1000, 1)

    opencv.imwrite('marker.png', img)

    opencv.imshow("marker", img)
    opencv.waitKey(0)


def locate_marker(marker_id, image_name):
    input_img = opencv.imread(image_name)
    #input_img = opencv.resize(input_img, (int(input_img.shape[1] / 5), int(input_img.shape[0] / 5)))

    corners, ids, rejected_img_points = opencv.aruco.detectMarkers(input_img, marker_dictionary)

    output = opencv.aruco.drawDetectedMarkers(input_img, corners, ids)

    for i in range(ids.shape[0]):
        if ids[i] == marker_id:
            for corner in corners[i][0]:
                opencv.circle(output, (corner[0], corner[1]), 4, (255, 0, 0), -1)

    opencv.imshow('marker', output)
    opencv.waitKey(10000)

    return corners, ids, rejected_img_points


if __name__ == '__main__':
    locate_marker(20, 'engg_m_test5.jpg')
