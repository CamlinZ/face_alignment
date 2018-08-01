"""
This script is slightly modified for facial landmark localization.
You can find the original code here:
https://github.com/opencv/opencv/blob/master/samples/dnn/resnet_ssd_face_python.py
"""
# -*- coding:utf-8 -*-

import cv2 as cv
from cv2 import dnn
import os
import shutil

WIDTH = 300
HEIGHT = 300

PROTOTXT = '/Users/camlin_z/Data/Project/OpenCV/samples/dnn/face_detector/deploy.prototxt'
MODEL = '/Users/camlin_z/Data/Project/OpenCV/samples/dnn/face_detector/deep-learning-face-detection/res10_300x300_ssd_iter_140000.caffemodel'

CASCADES_FILE = "/Users/camlin_z/Data/Project/OpenCV/data/lbpcascades/lbpcascade_frontalface_improved.xml"
CASCADES = cv.CascadeClassifier(CASCADES_FILE)

NET = dnn.readNetFromCaffe(PROTOTXT, MODEL)
img_dir = "/Users/camlin_z/Data/dataset/data/300W/01_Indoor/"
img_dst = "/Users/camlin_z/Data/dataset/data_test/"

def get_lbp_facebox(image):
    """
    Get the bounding box of faces in image by LBP feature.
    """
    rects = CASCADES.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                      flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    for rect in rects:
        rect[2] += rect[0]
        rect[3] += rect[1]
    return rects


def get_facebox(image=None, threshold=0.5):
    """
    Get the bounding box of faces in image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    confidences = []
    faceboxes = []

    NET.setInput(dnn.blobFromImage(
        image, 1.0, (WIDTH, HEIGHT), (104.0, 177.0, 123.0), False, False))
    detections = NET.forward()

    for result in detections[0, 0, :, :]:
        confidence = result[2]
        if confidence > threshold:
            x_left_bottom = int(result[3] * cols)
            y_left_bottom = int(result[4] * rows)
            x_right_top = int(result[5] * cols)
            y_right_top = int(result[6] * rows)
            confidences.append(confidence)
            faceboxes.append(
                [x_left_bottom, y_left_bottom, x_right_top, y_right_top])
    return confidences, faceboxes


def draw_result(image, confidences, faceboxes):
    """Draw the detection result on image"""
    for result in zip(confidences, faceboxes):
        conf = result[0]
        facebox = result[1]

        cv.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), (0, 255, 0))
        label = "face: %.4f" % conf
        label_size, base_line = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                     (facebox[0] + label_size[0],
                      facebox[1] + base_line),
                     (0, 255, 0), cv.FILLED)
        cv.putText(image, label, (facebox[0], facebox[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


def draw_box(image, faceboxes, box_color=(255, 255, 255)):
    """Draw square boxes on image"""
    for facebox in faceboxes:
        cv.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), box_color, 3)


def main():
    """The main entrance"""
    dataset = os.listdir(img_dir)
    for img_name in dataset:
        print img_name
        img = cv.imread(os.path.join(img_dir, img_name))
        confidences, faceboxes = get_facebox(img, threshold=0.5)
        # draw_result(img, confidences, faceboxes)
        # lbp_box = get_lbp_facebox(img)
        draw_box(img, faceboxes)
        # cv.imshow("detections", img)
        # if cv.waitKey(100) != -1:
        #     break
        cv.imwrite(img_dst + img_name, img)


if __name__ == '__main__':
    main()
