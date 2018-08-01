import os
import cv2
import subprocess
import numpy as np

image_format_list = ["jpeg", "jpg", "gif", "png", "bmp", "tiff", "ppm", "pgm", "pbm", "pnm"]

current_path = "/Users/camlin_z/Data/data/data3"
file_new_path = "/Users/camlin_z/Data/data/data3_test"

count = 0
total_files = 0

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 120, 120)

cv2.namedWindow("Converted", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Converted", 120, 120)

def convert_to_square():
    for file_path, _, file_names in os.walk(current_path, followlinks=False):
        for file_name in file_names:
            if file_name.split(".")[-1] in image_format_list:
                count += 1
                # file_name = "001.jpeg"
                # file_path = "/Users/camlin_z/Data/data/data4/"
                print(file_name)
                image = cv2.imread(os.path.join(file_path, file_name))
                height, width, depth = image.shape
                t_size = max([height, width])
                new_image = np.zeros([t_size, t_size, depth], np.uint8)+200

                if width > height:
                    # start = int((t_size - height)/2)
                    # new_image[start:start+height, 0:, 0:] = image
                    new_image[0:height, 0:, 0:] = image
                else:
                    # start = int((t_size - width)/2)
                    # new_image[0:, start:start + width, 0:] = image
                    new_image[0:, 0:width, 0:] = image

                if width != height:
                    cv2.imwrite(os.path.join(file_path, file_name), new_image)

                # cv2.imshow("Original", image)
                # cv2.imshow("Converted", new_image)
                #
                # cv2.waitKey(30)
    else:
        print("Not a image file: ", file_name)

    total_files += 1

    print("Total files: %2d, Images converted: %2d" % (total_files, count))

if __name__ == '__main__':
    convert_to_square()
