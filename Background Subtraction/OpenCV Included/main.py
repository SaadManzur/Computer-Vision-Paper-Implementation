import cv2 as opencv


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

    subtractor = opencv.createBackgroundSubtractorMOG2()
    subtractor.setDetectShadows(False)

    #print(subtractor.setHistory(100))

    while True:
        ret, frame = video.read()

        if frame is not None:
            foreground_mask = subtractor.apply(frame)

            opencv.imshow('frame', foreground_mask)

            key = opencv.waitKey(30) & 0xff
            if key == 'q':
                break

        else:
            break

    video.release()

    opencv.destroyAllWindows()


if __name__ == '__main__':
    subtract_background('resources/engg_m3.webm')
