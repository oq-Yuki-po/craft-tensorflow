import cv2
import numpy as np
import Polygon as plg
import tensorflow as tf


def watershed(original_image: np.ndarray, image: np.ndarray):

    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    sure_bg = mb

    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, surface_fg)

    ret, markers = cv2.connectedComponents(surface_fg)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
                                                                         connectivity=4)

    markers = labels.copy() + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(original_image, markers=markers)
    original_image[markers == -1] = [0, 0, 255]

    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)
