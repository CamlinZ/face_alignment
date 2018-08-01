# coding=utf-8

import cv2
import os
import shutil
from compiler.ast import flatten
import glob

img_dir = "/Users/camlin_z/Data/data_new/custom/"

txt_dir = "/Users/camlin_z/Data/data/landmark/testset_bbox/"
img_dir1 = "/Users/camlin_z/Data/68landmark/img/"

txt_dir1 = "/Users/camlin_z/Data/landmark/txt/"
img_dir2 = "/Users/camlin_z/Data/data_new/out/"


def change_name(img_dir, txt_dir1):
    # for file in os.listdir(txt_dir):
    #     if os.path.isfile(os.path.join(txt_dir, file)) == True:
    #         if file.find('.') < 0:
    #             newname = file + '.txt'
    #             os.rename(os.path.join(path, file), os.path.join(path, newname))
    #             print file, 'ok'

    for img_name in os.listdir(img_dir):
        # flag = img_name.find('.')
        pts_name = img_name + ".txt"
        if not os.path.exists(root + "data2_pts/" + pts_name):
            shutil.move(root + "data2/" )

def test_all(txt_dir, img_dir, img_dir1):
    for file in os.listdir(txt_dir):
        flag = file.find('.')
        img_name = file[:flag] + ".jpeg"
        print img_name
        if os.path.exists(os.path.join(txt_dir, file)) and os.path.exists(os.path.join(img_dir, img_name)) == True:
            lines = open(os.path.join(txt_dir, file))
            label = []
            for line in lines:
                label.append(line.strip().split())
            label = flatten(label)

            img = cv2.imread(img_dir + img_name)
            img_copy = img.copy()

            x1 = int(float(label[0]))
            y1 = int(float(label[1]))
            w = int(float(label[2]))
            h = int(float(label[3]))
            cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 5)
            for i in range(5, 139, 2):
                cv2.circle(img_copy, (int(label[i+1]), int(label[i])), 2, (255, 255, 255))
            cv2.imwrite(img_dir1 + img_name, img_copy)
        else:
            print img_name + "not exist!"

def test_bbox(txt_dir, img_dir, img_dir2):
    for file in os.listdir(txt_dir):
        flag = file.find('.')
        img_name = file[:flag] + ".jpg"
        print img_name
        if os.path.exists(os.path.join(txt_dir, file)) and os.path.exists(os.path.join(img_dir, img_name)) == True:
            lines = open(os.path.join(txt_dir, file))
            label = []
            for line in lines:
                label.append(line.strip().split())
            label = flatten(label)
            print label

            img = cv2.imread(img_dir + img_name)
            img_copy = img.copy()

            x1 = int(float(label[0]))
            y1 = int(float(label[1]))
            x2 = int(float(label[2]))
            y2 = int(float(label[3]))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 255, 255), 5)
            cv2.imwrite(img_dir2 + img_name, img_copy)
        else:
            print img_name + " not exist!"

def test_landmark(img_dir, img_dir2):
    for file in glob.glob(os.path.join(img_dir, "*.pts")):
        file = os.path.basename(file)
        flag = file.find('.')
        img_name = file[:flag] + ".jpeg"
        print img_name
        if os.path.exists(os.path.join(img_dir, file)) and os.path.exists(os.path.join(img_dir, img_name)) == True:
            label = []
            with open(os.path.join(img_dir, file)) as file:
                line_count = 0
                for line in file:
                    if "version" in line or "points" in line or "{" in line or "}" in line:
                        continue
                    else:
                        loc_x, loc_y = line.strip().split()
                        label.append([float(loc_x), float(loc_y)])
                        line_count += 1
            label = flatten(label)

            img = cv2.imread(img_dir + img_name)
            img_copy = img.copy()
            for i in range(0, 135, 2):
                cv2.circle(img_copy, (int(float(label[i])), int(float(label[i + 1]))), 2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.imwrite(img_dir2 + img_name, img_copy)
        else:
            print img_name + " not exist!"

def test_landmark_crop(txt_dir1, img_dir, img_dir2):
    for file in open(txt_dir1):
        flag = file.find(' ')
        img_name = file[:flag]
        imgname = os.path.basename(img_name)
        label = file[flag:].strip().split()
        print imgname
        if os.path.exists(txt_dir1) and os.path.exists(os.path.join(img_dir, imgname)) == True:
            img = cv2.imread(img_dir + imgname)
            img_copy = img.copy()
            for i in range(0, 136, 2):
                cv2.circle(img_copy, (int(float(label[i])), int(float(label[i + 1]))), 2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.imwrite(img_dir2 + imgname, img_copy)
        else:
            print img_name + "not exist!"
        cv2.imshow("face", img_copy)
        if cv2.waitKey(10) == 27:
            cv2.waitKey()

if __name__ == '__main__':
    # change_name(txt_dir)
    # test_all(txt_dir, img_dir, img_dir1)
    # test_bbox(txt_dir, img_dir, img_dir2)
    test_landmark(img_dir, img_dir2)
    # test_landmark_crop(txt_dir1, img_dir, img_dir2)