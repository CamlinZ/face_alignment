# -*- coding:utf-8 -*-

import os

txt_dir = "/Users/camlin_z/Data/data_fine/label_test.txt"
txt_new_dir = "/Users/camlin_z/Data/1/data/label_new_val.txt"
img_dir = "/Users/camlin_z/Data/data_fine/"

def clear_data():
    # file_new = open(txt_new_dir, 'w+')
    with open(txt_dir) as file:
        for line in file:
            img_name = line.strip().split()[0]
            print img_name
            if not os.path.exists(os.path.join(img_dir, img_name)):
                print img_name

if __name__ == '__main__':
    clear_data()