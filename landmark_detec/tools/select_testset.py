import os
import shutil

test_sum = 400
label_string = "data2"

# label_path = "/Users/camlin_z/Data/label_train/label_train_" + label_string + ".txt"
# label_test = "/Users/camlin_z/Data/label_val/label_val_" + label_string + ".txt"
# label_train = "/Users/camlin_z/Data/label_train_/label_train_" + label_string + ".txt"
#
# root = "/Users/camlin_z/Data/data_train"
# test_dir = "/Users/camlin_z/Data/data_val/"

label_path = "/Users/camlin_z/Data/data/data/data4.txt"
label_test = "/Users/camlin_z/Data/data/data/label_test_data2.txt"
label_val = "/Users/camlin_z/Data/data/data/label_val_data2.txt"

root = "/Users/camlin_z/Data/data/data"
test_dir = "/Users/camlin_z/Data/data/data/test/"
val_dir = "/Users/camlin_z/Data/data/data/val/"

def test():
    test_label = open(label_test, 'w+')
    train_label = open(label_val, 'w+')
    with open(label_path) as lines:
        flag = 0
        for line in lines:
            if flag < test_sum:
                test_label.write(line)
                # img_name = line.split()[0]
                # shutil.copy(os.path.join(root, img_name), test_dir)
                flag += 1
            else:
                train_label.write(line)
                # img_name = line.split()[0]
                # shutil.copy(os.path.join(root, img_name), val_dir)

if __name__ == '__main__':
    test()