from __future__ import division
import os
import cv2
from compiler.ast import flatten

txt_dir = "/Users/camlin_z/Data/68landmark/txt/"
txt_new_dir = "/Users/camlin_z/Data/68landmark/landmark/"

def trans_label():
    files = os.listdir(txt_dir)
    for file in files:
        flag = file.find(".")
        if flag > 0:
            txt_name = file[:flag] + ".pts"
            print txt_name
            line = open(txt_dir + file, 'r')
            for label in line:
                label = label.strip().split()
                label = map(float, label)
                file_new = open(txt_new_dir + txt_name, 'w+')
                file_new.write("version: 1" + "\n")
                file_new.write("n_points: 68" + "\n")
                file_new.write("{" + "\n")
                for i in range(0, 135, 2):
                    file_new.write(str(label[i]) + " " + str(label[i+1]) + "\n")
                file_new.write("}")
        else:
            print file, " not exist!"

if __name__ == '__main__':
    trans_label()