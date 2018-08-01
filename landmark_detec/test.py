# coding=utf-8

import numpy as np
import cv2
import caffe
from PIL import Image, ImageDraw
import time
import os

blobname = "68point"
feature_dim = 68
normalization = "centers"

root = "/Users/camlin_z/Data/Project/caffe-68landmark/landmark_detec/"
deploy = root + "deploy.prototxt"
# caffe_model = root + "snapshot_all1/snapshot_iter_250000.caffemodel"
# caffe_model = root + "snapshot_part1/oldfinetune_iter_20000.caffemodel"
caffe_model = root + "init.caffemodel"
# caffe_model = root + "snapshot2/fine_iter_400000.caffemodel"
# caffe_model = root + "snapshot/final_iter_150000.caffemodel"
img_dir = "/Users/camlin_z/Data/data_fine/"
img_dir_out = "/Users/camlin_z/Data/data_fine/out/"
label_file = "/Users/camlin_z/Data/data_fine/label_test.txt"
img_path = "/Users/camlin_z/Data/data_fine/data2/2415.jpeg"

net = caffe.Net(deploy, caffe_model, caffe.TEST)
caffe.set_mode_cpu()

def compute_normDist(landmark_gt):
    # compute the normalized distance
    normDist = 1
    # compute the normalized distance
    if normalization == 'centers':
        normDist = np.linalg.norm(
            np.mean(landmark_gt[36:42], axis=0) - np.mean(landmark_gt[42:48], axis=0))
    elif normalization == '194centers':
        normDist = np.linalg.norm(
            np.mean(landmark_gt[134:154], axis=0) - np.mean(landmark_gt[114:134], axis=0))
    elif normalization == 'corners':
        normDist = np.linalg.norm(landmark_gt[36] - landmark_gt[45])
    elif normalization == 'diagonal':
        height, width = np.max(landmark_gt, axis=0) - np.min(landmark_gt, axis=0)
        normDist = np.sqrt(width ** 2 + height ** 2)
    return normDist

def detec_whole(img_dir, img_dir_out, label_file):
    time_sum = 0
    mser_sum = 0
    mser_norm_sum = 0
    id_sum = 0

    fid = open(label_file, 'r')
    for id in fid:
        id_sum += 1
        flag = id.find(' ')
        # 读取文件中的标签信息
        image_name = id[:flag]
        label_true = id[flag:]
        label_true = label_true.strip().split()

        landmark_gt = []
        for i in range(0, 135, 2):
            landmark_gt.append([int(label_true[i]), int(label_true[i + 1])])
        # print landmark_gt
        normDist = compute_normDist(landmark_gt)
        print "normDist：", normDist

        label_true = map(float, label_true)
        label_true = np.array(label_true, np.float32)
        imgname = os.path.basename(image_name)
        print imgname
        # print label_true

        img = cv2.imread(img_dir + image_name)
        img_draw = img.copy()
        sh = img.shape
        h = sh[0]
        w = sh[1]
        rw = (w + 1) / 2
        rh = (h + 1) / 2

        # 以下网络输出了预测的68点坐标
        # start = time.time()
        img = np.array(img, np.float32)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        start = time.time()
        out = net.forward()
        landmark = out[blobname]
        elap = time.time() - start
        landmark = np.array(landmark, np.float32)
        landmark[0: 136: 2] = (landmark[0: 136: 2] * rh) + rh
        landmark[1: 136: 2] = (landmark[1: 136: 2] * rw) + rw
        # print landmark
        time_sum += elap
        print "time:", elap

        for i in range(0, 136, 2):
            cv2.circle(img_draw, (int(landmark[0][i]), int(landmark[0][i + 1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.imwrite(img_dir_out + imgname, img_draw)

        # print label_true
        # print landmark
        # print landmark_gt

        v = label_true - landmark
        v = v*v
        v = v[0][0::2] + v[0][1:: 2]
        sv = np.power(v, 0.5)
        mser = sum(sv) / feature_dim
        mser_norm = mser / normDist

        mser_sum += mser
        mser_norm_sum += mser_norm
        print "mser:", mser
        print "normalized distance: ", mser_norm

    print "Average time:", time_sum/id_sum
    print "Average mser:", mser_sum/id_sum
    print "Average normalized distance: ", mser_norm_sum / id_sum

def detec_single():
    img = cv2.imread(img_path)
    # draw = ImageDraw.Draw(img1)
    sh = img.shape
    print sh
    h = sh[0]
    w = sh[1]
    rw = (w + 1)/2
    rh = (h + 1)/2
    img = np.array(img, np.float32)
    img_copy = img.copy()

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))
    net.blobs['data'].data[...] = transformer.preprocess('data',img)
    start = time.time()
    out = net.forward()
    elap = time.time() - start
    print "time:", elap
    landmark = out[blobname]
    landmark = np.array(landmark, np.float32)
    landmark[0: 136: 2] = (landmark[0: 136: 2] * rh ) + rh
    landmark[1: 136: 2] = (landmark[1: 136: 2] * rw ) + rw
    # print landmark

    for i in range(0, 136, 2):
        cv2.circle(img_copy, (int(landmark[0][i]), int(landmark[0][i+1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
    # draw.point(landmark[0], (225, 225, 255))
    # del draw
    cv2.imwrite(root + 'test.jpg', img_copy)
    # img1.show()

if __name__ == '__main__':
    detec_whole(img_dir, img_dir_out, label_file)
    # detec_single()