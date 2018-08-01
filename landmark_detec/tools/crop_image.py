# coding=utf-8

from __future__ import division
import cv2
import os
import numpy as np
from compiler.ast import flatten
from PIL import Image

SIZE = 224
file_string = "data4/"

img_dir = "/Users/camlin_z/Data/data/data/data4/"
txt_bbox = "/Users/camlin_z/Data/data/data/data4_label/"
txt_pts = "/Users/camlin_z/Data/data/pts/data4/"
txt_label = "/Users/camlin_z/Data/data/data/data4.txt"
imgcrop_dir = "/Users/camlin_z/Data/data/data/data4_/"

txt_new = "/Users/camlin_z/Data/label/data3_new.txt"
img_resize_dir = "/Users/camlin_z/Data/data/"
img_resize_dir_dst = "/Users/camlin_z/Data/data/data3_new/"

def crop(img_dir, txt_pts, txt_bbox, imgcrop_dir, txt_label):
    label_txt = open(txt_label, 'w+')
    # 根据特征点坐标文件夹作为基准序号
    for file in os.listdir(txt_pts):
        # 分别找到图像和检测框的名字位置
        flag = file.find('t')
        flag -= 1
        flagb = file.find('j')
        flagb -= 1
        img_name = file[:flag]
        txt_bbox_name = file[:flagb] + ".pts"
        print img_name
        print txt_bbox_name
        if os.path.exists(os.path.join(img_dir, img_name)) and os.path.exists(os.path.join(txt_bbox, txt_bbox_name)) == True:
            # 取出特征点文件夹中的所有值
            lines_pts = open(os.path.join(txt_pts, file))
            label_pts = []
            for line_pts in lines_pts:
                label_pts.append(line_pts.strip().split())
            label_pts = flatten(label_pts)
            # 注意一定要将字符型list转换为int型再进行比较，否则下面求出的不是整数的最大最小值
            label_pts = map(float, label_pts)
            # 找出所有特征点中的最大x和y值
            max_x = max(label_pts[::2])
            min_x = min(label_pts[::2])
            max_y = max(label_pts[1::2])
            min_y = min(label_pts[1::2])
            print min_x, min_y, max_x, max_y

            # 取出所有的监测框的值
            lines_bbox = open(os.path.join(txt_bbox, txt_bbox_name))
            label_bbox = []
            for line_bbox in lines_bbox:
                label_bbox.append(line_bbox.strip().split())
            label_bbox = flatten(label_bbox)
            x1 = float(label_bbox[0])
            y1 = float(label_bbox[1])
            W = float(label_bbox[2])
            H = float(label_bbox[3])
            x2 = x1 + W
            y2 = y1 + H
            print x1, y1, x2, y2

            # 首先判断点和框的位置是否吻合(只要有10个点在框内就认为两者吻合)
            k = 0
            for x in label_pts[::2]:
                if x < x2 and x > x1:
                    k +=1

            img = cv2.imread(img_dir + img_name)
            (h, w, ch) = img.shape
            # print w, h
            # 框和点的位置吻合，就采取下面的逻辑
            if k >= 6:
                # 将以上得到的坐标和68个特征点的最大值坐标进行比较，如果存在小于其最大值的情况，将该处坐标增加最大值+(h-w)/2
                if max_x > x2:
                    x2 = min(max_x+((h-w)/2), w)
                if max_y > y2:
                    y2 = min(max_y+((h-w)/2), h)
                if min_x < x1:
                    x1 = max(min_x-((h-w)/2), 0)
                if min_y < y1:
                    y1 = max(min_y-((h-w)/2), 0)
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, w)
                    y2 = min(y2, h)
            else:
                x1 = max(min_x - ((h - w) / 2), 0)
                y1 = max(min_y - ((h - w) / 2), 0)
                x2 = min(max_x + ((h - w) / 2), w)
                y2 = min(max_y + ((h - w) / 2), h)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            print x1, y1, x2, y2
            # 剪裁后产生的crop图像的长宽
            crop_w = x2 - x1
            crop_h = y2 - y1

            img_copy = img.copy()
            img_copy = np.array(img_copy, np.float32)
            img_crop = img_copy[y1:y2, x1:x2]

            if crop_h != SIZE or crop_w != SIZE:
                img_crop = cv2.resize(img_crop, (SIZE, SIZE))
                ratio_w = SIZE / crop_w
                ratio_h = SIZE / crop_h
                cv2.imwrite(imgcrop_dir + img_name, img_crop)
            else:
                ratio_w = 1
                ratio_h = 1
                cv2.imwrite(imgcrop_dir + img_name, img_crop)

            # cv2.imwrite(imgcrop_dir + img_name, img_crop)

            label_txt.write(file_string + img_name + " ")
            for i in range(0, 135, 2):
                if i != 134:
                    x_temp = int((label_pts[i] - x1) * ratio_w)
                    y_temp = int((label_pts[i+1] - y1) * ratio_h)
                    label_txt.write(str(x_temp) + " " + str(y_temp) + " ")
                else:
                    x_temp = int((label_pts[i] - x1) * ratio_w)
                    y_temp = int((label_pts[i + 1] - y1) * ratio_h)
                    label_txt.write(str(x_temp) + " " + str(y_temp))
            label_txt.write("\n")

        else:
            print img_name + "not exist!"


def resize_crop(txt_label, txt_new, img_resize_dir, img_resize_dir_dst):
    file = open(txt_label)
    file_new = open(txt_new, 'w')
    for line in file:
        flag = line.find(' ')
        img_name = line[:flag]
        imgname = os.path.basename(img_name)
        print imgname
        label = line[flag:].strip().split()
        label = map(int, label)

        # img = Image.open(img_resize_dir + img_name)
        # (w, h) = img.size
        img = cv2.imread(img_resize_dir + img_name)
        img_copy = img.copy()
        (w, h, c) = img.shape

        if h != SIZE or w != SIZE:
            # if h < w:
            #     img_resize = img.resize((w * SIZE / h, SIZE), Image.ANTIALIAS)
            #     label = [(i * SIZE / h) for i in label]
            # else:
            #     img_resize = img.resize((SIZE, h * SIZE / w), Image.ANTIALIAS)
            #     label = [(i * SIZE / h) for i in label]
            # img_resize.save(img_resize_dir_dst + imgname, format="jpeg")
            img_copy = cv2.resize(img_copy, (SIZE, SIZE))
            ratio_w = SIZE / w
            ratio_h = SIZE / h
            cv2.imwrite(img_resize_dir_dst + imgname, img_copy)

        # file_new.write(img_name + " ")
        # for i in range(0, 135, 2):
        #     if i != 134:
        #         x_temp = int(label[i])
        #         y_temp = int(label[i + 1])
        #         file_new.write(str(x_temp) + " " + str(y_temp) + " ")
        #     else:
        #         x_temp = int(label[i])
        #         y_temp = int(label[i + 1])
        #         file_new.write(str(x_temp) + " " + str(y_temp))
        # file_new.write("\n")
        file_new.write(img_name + " ")
        for i in range(0, 135, 2):
            if i != 134:
                x_temp = int(label[i] * ratio_w)
                y_temp = int(label[i + 1] * ratio_h)
                file_new.write(str(x_temp) + " " + str(y_temp) + " ")
            else:
                x_temp = int(label[i] * ratio_w)
                y_temp = int(label[i + 1] * ratio_h)
                file_new.write(str(x_temp) + " " + str(y_temp))
        file_new.write("\n")

def test_bbox(img_dir, txt_bbox, imgcrop_dir):
    for file in os.listdir(txt_bbox):
        flag = file.find('.')
        img_name = file[:flag] + ".jpeg"
        print img_name
        if os.path.exists(os.path.join(txt_bbox, file)) and os.path.exists(os.path.join(img_dir, img_name)) == True:
            lines = open(os.path.join(txt_bbox, file))
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
            x2 = x1 + w
            y2 = y1 + h
            img_copy = np.array(img_copy, np.float32)
            img_crop = img_copy[y1:y2, x1:x2]
            cv2.imwrite(imgcrop_dir + img_name, img_crop)
            cv2.imwrite(img_dir1 + img_name, img_copy)
        else:
            print img_name + "not exist!"

if __name__ == '__main__':
    crop(img_dir, txt_pts, txt_bbox, imgcrop_dir, txt_label)
    # resize_crop(txt_label, txt_new, img_resize_dir, img_resize_dir_dst)