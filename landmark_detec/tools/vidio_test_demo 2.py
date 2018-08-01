# -*- coding: utf-8 -*-
"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""
from __future__ import division
import caffe
import os
import cv2
import numpy as np
import face_detector_image as fd
from compiler.ast import flatten
import shutil
import time

# 68点测试代码
feature_dim = 68
root = "/Users/camlin_z/Data/526_caffe/"
vidio_path = root + "vid.avi"
gt_landmark_path = root + 'annot/'
out_dir = root + "out/"

root = "/Users/camlin_z/Data/Project/caffe-68landmark/landmark_detec/"
deploy = root + "deploy.prototxt"
caffe_model = root + "init.caffemodel"
normalization = "centers"

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

def read_points(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split()
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points

def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)
    return image

def points_are_valid(points, image):
    """Check if all points are in image"""
    min_box = get_minimal_box(points)
    if box_in_image(min_box, image):
        return True
    return False

def get_square_box(box):
    """Get the square boxes which are ready for CNN from the boxes"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def get_minimal_box(points):
    """
    Get the minimal bounding box of a group of points.
    The coordinates are also converted to int numbers.
    """
    min_x = int(min([point[0] for point in points]))
    max_x = int(max([point[0] for point in points]))
    min_y = int(min([point[1] for point in points]))
    max_y = int(max([point[1] for point in points]))
    return [min_x, min_y, max_x, max_y]

def move_box(box, offset):
    """Move the box to direction specified by offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]

def expand_box(square_box, scale_ratio=1.2):
    """Scale up the box"""
    assert (scale_ratio >= 1), "Scale ratio should be greater than 1."
    delta = int((square_box[2] - square_box[0]) * (scale_ratio - 1) / 2)
    left_x = square_box[0] - delta
    left_y = square_box[1] - delta
    right_x = square_box[2] + delta
    right_y = square_box[3] + delta
    return [left_x, left_y, right_x, right_y]

def points_in_box(points, box):
    """Check if box contains all the points"""
    minimal_box = get_minimal_box(points)
    return box[0] <= minimal_box[0] and \
           box[1] <= minimal_box[1] and \
           box[2] >= minimal_box[2] and \
           box[3] >= minimal_box[3]

def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

def box_is_valid(image, points, box):
    """Check if box is valid."""
    # Box contains all the points.
    points_is_in_box = points_in_box(points, box)

    # Box is in image.
    box_is_in_image = box_in_image(box, image)

    # Box is square.
    w_equal_h = (box[2] - box[0]) == (box[3] - box[1])

    # Return the result.
    return box_is_in_image and points_is_in_box and w_equal_h

def fit_by_shifting(box, rows, cols):
    """Method 1: Try to move the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # Check if moving is possible.
    if right_x - left_x <= cols and bottom_y - top_y <= rows:
        if left_x < 0:  # left edge crossed, move right.
            right_x += abs(left_x)
            left_x = 0
        if right_x > cols:  # right edge crossed, move left.
            left_x -= (right_x - cols)
            right_x = cols
        if top_y < 0:  # top edge crossed, move down.
            bottom_y += abs(top_y)
            top_y = 0
        if bottom_y > rows:  # bottom edge crossed, move up.
            top_y -= (bottom_y - rows)
            bottom_y = rows

    return [left_x, top_y, right_x, bottom_y]

def fit_by_shrinking(box, rows, cols):
    """Method 2: Try to shrink the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # The first step would be get the interlaced area.
    if left_x < 0:  # left edge crossed, set zero.
        left_x = 0
    if right_x > cols:  # right edge crossed, set max.
        right_x = cols
    if top_y < 0:  # top edge crossed, set zero.
        top_y = 0
    if bottom_y > rows:  # bottom edge crossed, set max.
        bottom_y = rows

    # Then found out which is larger: the width or height. This will
    # be used to decide in which dimention the size would be shrinked.
    width = right_x - left_x
    height = bottom_y - top_y
    delta = abs(width - height)
    # Find out which dimention should be altered.
    if width > height:  # x should be altered.
        if left_x != 0 and right_x != cols:  # shrink from center.
            left_x += int(delta / 2)
            right_x -= int(delta / 2) + delta % 2
        elif left_x == 0:  # shrink from right.
            right_x -= delta
        else:  # shrink from left.
            left_x += delta
    else:  # y should be altered.
        if top_y != 0 and bottom_y != rows:  # shrink from center.
            top_y += int(delta / 2) + delta % 2
            bottom_y -= int(delta / 2)
        elif top_y == 0:  # shrink from bottom.
            bottom_y -= delta
        else:  # shrink from top.
            top_y += delta

    return [left_x, top_y, right_x, bottom_y]

def fit_box(box, image, points):
    """
    Try to fit the box, make sure it satisfy following conditions:
    - A square.
    - Inside the image.
    - Contains all the points.
    If all above failed, return None.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    # First try to move the box.
    box_moved = fit_by_shifting(box, rows, cols)

    # If moving faild ,try to shrink.
    if box_is_valid(image, points, box_moved):
        return box_moved
    else:
        box_shrinked = fit_by_shrinking(box, rows, cols)

    # If shrink failed, return None
    if box_is_valid(image, points, box_shrinked):
        return box_shrinked

    # Finally, Worst situation.
    print("Fitting failed!")
    return None

def get_valid_box(image, points):
    """
    Try to get a valid face box which meets the requirments.
    The function follows these steps:
        1. Try method 1, if failed:
        2. Try method 0, if failed:
        3. Return None
    """

    # Try method 1 first.
    def _get_postive_box(raw_boxes, points):
        for box in raw_boxes:
            # Move box down.
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = move_box(box, [0, offset_y])

            # Make box square.
            square_box = get_square_box(box_moved)

            # Remove false positive boxes.
            if points_in_box(points, square_box):
                return square_box
        return None

    # Try to get a positive box from face detection results.
    _, raw_boxes = fd.get_facebox(image, threshold=0.5)
    positive_box = _get_postive_box(raw_boxes, points)
    if positive_box is not None:
        if box_in_image(positive_box, image) is True:
            return positive_box
        return fit_box(positive_box, image, points)

    # Method 1 failed, Method 0
    min_box = get_minimal_box(points)
    sqr_box = get_square_box(min_box)
    epd_box = expand_box(sqr_box)
    if box_in_image(epd_box, image) is True:
        return epd_box
    return fit_box(epd_box, image, points)

def get_new_pts(facebox, raw_points, ratio_w, ratio_h):
    """
    generate a new pts file according to face box

    """
    x = facebox[0]
    y = facebox[1]
    # print x, y
    new_point = []

    for label_pts in raw_points:
        x_temp = int(label_pts[0] / ratio_w + x)
        y_temp = int(label_pts[1] / ratio_h + y)
        new_point.append([x_temp, y_temp])

    return new_point

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

def compute_mesr(landmark_gt, landmark_pre, normDist):
    landmark_gt = flatten(landmark_gt)
    landmark_gt = map(int, landmark_gt)
    landmark_pre = flatten(landmark_pre)
    landmark_pre = map(int, landmark_pre)

    landmark_pre = np.array(landmark_pre, np.float32)
    landmark_gt = np.array(landmark_gt, np.float32)

    v = landmark_gt - landmark_pre
    v = v * v
    v = v[0::2] + v[1::2]
    sv = np.power(v, 0.5)

    mser = sum(sv) / feature_dim
    mser_norm = mser / normDist

    return mser, mser_norm

# turn img to vidio
def img_to_vedio():
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(out_dir + 'saveVideo.avi', fourcc, fps, 3, (1280, 720))

    img = []
    img_path = out_dir
    for item in os.listdir(img_path):
        img.append(os.path.join(img_path, item))
    img.sort()

    for i in range(len(img)):
        print img_path + str(i) + '.png'
        frame = cv2.imread(img_path + str(i) + '.png')
        videoWriter.write(frame)
    videoWriter.release()

def preview(img, point_file):
    """
    Preview points on image.
    """
    # Read the points from file.
    raw_points = read_points(point_file)

    # Safe guard, make sure point importing goes well.
    assert len(raw_points) == 68, "The landmarks should contain 68 points."

    # Fast check: all points are in image.
    if points_are_valid(raw_points, img) is False:
        return None

    # Get the valid facebox.
    facebox = get_valid_box(img, raw_points)
    if facebox is None:
        print("Using minimal box.")
        facebox = get_minimal_box(raw_points)

    # Extract valid image area.
    face_crop = img[facebox[1]:facebox[3],
                facebox[0]: facebox[2]]

    rw = 1
    rh = 1
    # Check if resize is needed.
    width = facebox[2] - facebox[0]
    height = facebox[3] - facebox[1]
    if width != height:
        print('opps!', width, height)
    if (width != 224) or (height != 224):
        face_crop = cv2.resize(face_crop, (224, 224))
        rw = 224 / width
        rh = 224 / height

    return raw_points, face_crop, facebox, rw, rh

def main():
    """
    The main entrance
    """
    # load caffe model
    net = caffe.Net(deploy, caffe_model, caffe.TEST)
    caffe.set_mode_cpu()

    mkr(out_dir)

    # 测试数据初始化
    time_sum = 0
    mser_sum = 0
    mser_norm_sum = 0
    id_sum = 0
    normDist_sum = 0

    # load ground truth list
    gt_landmark = []
    for item in os.listdir(gt_landmark_path):
        gt_landmark.append(os.path.join(gt_landmark_path, item))
    gt_landmark.sort()

    # load vidio
    vidIn = cv2.VideoCapture(vidio_path)

    # read every frame of vidio and gt
    for landmark_of_this_img in gt_landmark:

        id_sum += 1
        img_raw = vidIn.read()[1]
        # generate crop img and new pts
        landmark_gt, img, facebox, rate_w, rate_h = preview(img_raw, landmark_of_this_img)
        normDist = compute_normDist(landmark_gt)
        print normDist

        assert len(landmark_gt) == feature_dim, "The landmarks should contain 68 points."

        # feed crop img into net and generate 68 landmarks
        sh = img.shape
        h = sh[0]
        w = sh[1]
        rw = (w + 1) / 2
        rh = (h + 1) / 2
        img = np.array(img, np.float32)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([127.5, 127.5, 127.5]))
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        start = time.time()
        out = net.forward()
        elap = time.time() - start
        time_sum += elap
        print "time:", elap
        landmark = out["68point"]
        landmark = np.array(landmark, np.float32)
        landmark[0: 136: 2] = (landmark[0: 136: 2] * rh) + rh
        landmark[1: 136: 2] = (landmark[1: 136: 2] * rw) + rw
        landmark = landmark.tolist()
        landmark = flatten(landmark)

        landmark_pre = []
        for i in range(0, 135, 2):
            landmark_pre.append([int(landmark[i]), int(landmark[i + 1])])

        landmark_pre_ori = get_new_pts(facebox, landmark_pre, rate_w, rate_h)

        # for point in landmark_gt:
        #     cv2.circle(img_raw, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1, cv2.LINE_AA)
        for point in landmark_pre_ori:
            cv2.circle(img_raw, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)

        # cv2.imshow("vedio", img_raw)
        cv2.imwrite(out_dir + str(id_sum) + '.png', img_raw)

        mser, mser_norm = compute_mesr(landmark_gt, landmark_pre_ori, normDist)

        normDist_sum += normDist
        mser_sum += mser
        mser_norm_sum += mser_norm
        print "mser:", mser
        print "normalized distance: ", mser_norm
        print "\n"

    # print "num_img: ", id_sum
    print "Average normDist: ", normDist_sum / id_sum
    print "Average time:", time_sum / id_sum
    print "Average mser:", mser_sum / id_sum
    print "Average normalized distance: ", mser_norm_sum / id_sum

    # img_to_vedio()


if __name__ == "__main__":
    main()
