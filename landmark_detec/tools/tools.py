# coding=utf-8

import cv2
from PIL import Image
from compiler.ast import flatten

root = "/Users/camlin_z/Data/Project/caffe-master-multilabel-normalize-randcrop-newloss/examples/landmark_detec/"
label_file = root + "label.txt"
img_dir = root + "data2/"
img_dir1 = root + "dataset_ver/"

# fid = open(label_file, 'r')
# for id in fid:
#     img_id = id.strip().split(' ')
#     img = cv2.imread(img_dir + img_id[0])
#     img1 = img.copy()
#     label = img_id[1:]
#     for i in range(0, 136, 2):
#         cv2.circle(img1, (int(float(label[i])), int(float(label[i+1]))), 3, (0, 0, 0))
#     cv2.imwrite(img_dir1 + img_id[0][5:], img1)

lines_pts = open("/Users/camlin_z/Data/dataset/data/data/data4_pts/052.jpeg.txt")
label_pts = []
for line_pts in lines_pts:
    label_pts.append(line_pts.strip().split())
label_pts = flatten(label_pts)
label_pts = map(int, label_pts)
# 找出所有特征点中的最大x和y值
max_x = int(float(max(label_pts[::2])))
print label_pts[0:136:2]
min_x = int(float(min(label_pts[::2])))
print label_pts[0:136:2]
max_y = int(float(max(label_pts[1::2])))
print label_pts[1:136:2]
min_y = int(float(min(label_pts[1::2])))
print label_pts[1:136:2]
print min_x, max_x, min_y, max_y