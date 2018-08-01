# coding=utf-8

import cv2
import os

bboxtxt_file = "/Users/camlin_z/Data/CelebA/list_bbox_celeba.txt"
lmtxt_file = "/Users/camlin_z/Data/CelebA/list_landmark_celeba.txt"
img_dir = "/Users/camlin_z/Data/CelebA/celeba/"
testimg_dir1 = "/Users/camlin_z/Data/CelebA/testimg/"
testimg_dir2 = "/Users/camlin_z/Data/CelebA/testimg1/"

def test_bboxtxt(img_dir, bboxtxt_file):
    if os.path.exists(testimg_dir1) and os.path.exists(img_dir):
        lines = open(bboxtxt_file)
        for line in lines:
            bbox = line.strip().split()
            img_name = bbox[0]
            x1 = int(bbox[1])
            y1 = int(bbox[2])
            w = int(bbox[3])
            h = int(bbox[4])
            img = cv2.imread(img_dir + img_name)
            img_copy = img.copy()
            cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 5)
            cv2.imwrite(testimg_dir1 + img_name, img_copy)

def test_landmark(img_dir, lmtxt_file):
    if os.path.exists(img_dir) and os.path.exists(testimg_dir2):
        lines = open(lmtxt_file)
        for line in lines:
            landmark = line.strip().split()
            img_name = landmark[0]
            print img_name
            img = cv2.imread(img_dir + img_name)
            img_copy = img.copy()
            for i in range(1, 136, 2):
                cv2.circle(img_copy, (int(float(landmark[i])), int(float(landmark[i + 1]))), 2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.imwrite(testimg_dir2 + img_name, img_copy)

if __name__ == '__main__':
    # test_bboxtxt(img_dir, bboxtxt_file)
    test_landmark(img_dir, lmtxt_file)