import cv2 as opencv
import numpy as np


def subtract_background(filename):
    video = opencv.VideoCapture(filename)

    while not video.isOpened():
        video = opencv.VideoCapture(filename)

        if opencv.waitKey(1000) == 'q':
            break

        print("Attempting again.")

    print('Frame count = ', video.get(opencv.CAP_PROP_FRAME_COUNT))
    print('Frame width = ', video.get(opencv.CAP_PROP_FRAME_WIDTH))
    print('Frame height = ', video.get(opencv.CAP_PROP_FRAME_HEIGHT))

    subtractor = opencv.createBackgroundSubtractorMOG2(detectShadows=False)

    # print(subtractor.setHistory(100))

    kernel = np.ones((9, 9), np.uint8)

    rects = []
    while True:
        ret, frame = video.read()

        if frame is not None:
            foreground_mask = subtractor.apply(frame)

            foreground_mask = opencv.morphologyEx(foreground_mask, opencv.MORPH_OPEN, np.ones((3, 3), np.uint8))
            foreground_mask = opencv.morphologyEx(foreground_mask, opencv.MORPH_CLOSE, kernel)

            im2, contours, hierarchy = opencv.findContours(foreground_mask, mode=opencv.CHAIN_APPROX_SIMPLE,
                                                           method=opencv.RETR_FLOODFILL)

            frame_contours = []
            for c in contours:
                if opencv.contourArea(c) > 300:
                    x, y, w, h = opencv.boundingRect(c)

                    if h > w:
                        frame_contours.append(np.array([x, y, w, h]))
                        opencv.rectangle(frame, (x, y), (x+w, y+h), color=(255, 0, 0))

            rects.append(frame_contours)

            opencv.imshow('frame', frame)

            key = opencv.waitKey(30) & 0xff
            if key == 'q':
                break

        else:
            break

    video.release()

    opencv.destroyAllWindows()


if __name__ == '__main__':
    subtract_background('resources/engg_m3.webm')
