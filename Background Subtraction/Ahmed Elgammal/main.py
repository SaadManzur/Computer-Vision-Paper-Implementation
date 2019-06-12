"""
Implementation of the paper "https://link.springer.com/chapter/10.1007/3-540-45053-X_48"
Title: Non-parametric Model for Background Subtraction
Python: 3.7
OpenCV: 4.1.0
OpenCV-contrib: 4.1.0.25
"""

import cv2 as opencv
import numpy as np
from matplotlib import pyplot as plt


def get_video_frames(filename):
    video = opencv.VideoCapture(filename)

    if not video.isOpened():
        print("Unable to open video file.")
        return

    original_frames = []

    while True:
        ret, frame = video.read()

        if frame is None:
            break

        original_frames.append(frame)

    return np.array(original_frames)


def gaussian_kernel_estimator(xt, xi, cov):
    d = xt.shape[0]
    nominator = np.exp(-0.5 * (xt - xi).T @ cov @ (xt - xi))
    denominator = (2 * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5

    return nominator / denominator


def estimate_covariance(sample):

    if sample.shape[0] == 0:
        cov = np.zeros((3, 3))
        np.fill_diagonal(cov, 1)
        return cov

    deviations = np.diff(sample, axis=0)

    '''
    deviations = np.zeros((sample.shape[0] - 1, sample.shape[1]))
    for i in range(sample.shape[0] - 1):
        deviations[i] = sample[i + 1, np.newaxis] - sample[i, np.newaxis]
    '''

    median = np.median(deviations, axis=0)
    sigma = (median / (0.68 * np.sqrt(2))) ** 2
    sigma += 1e-9

    d = sample[0].shape[0]
    cov = np.zeros((d, d))
    [row, col] = np.diag_indices(d)
    cov[row, col] = sigma

    return cov


def estimate_density(xt, sample):
    accumulator = 0
    cov = estimate_covariance(sample)

    for xi in sample:
        d = xt.shape[0]
        accumulator += gaussian_kernel_estimator(xt.reshape((d, 1)), xi.reshape((d, 1)), cov)

    return accumulator / sample.shape[0]


def get_pixel_probability(frame, previous_frames):
    probabilities = np.zeros(frame.shape)
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            probabilities[x, y] = estimate_density(frame[x, y], previous_frames[:, x, y])

    return probabilities


def process_frames(frames, threshold=0.5, sample_count=8):

    new_frames = np.zeros(frames.shape)

    for t in range(sample_count, frames.shape[0]):
        print(t)
        probability = get_pixel_probability(frames[t], frames[np.max(t-sample_count, 0):t])
        foreground_indices = probability < threshold
        new_frames[t, foreground_indices] = 1.0

    return new_frames


if __name__ == '__main__':
    frames = get_video_frames('resources/vtest.avi')
    new_frames = process_frames(frames)
    print(new_frames.shape)
