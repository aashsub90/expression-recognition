# -*- coding: utf-8 -*-

import os
import glob
import csv
import cv2


def make_dir_if_not_exist(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        print("Creation of the directory %s failed" % dirname)
    else:
        print("Successfully created the directory %s " % dirname)


def parse_UMD(image_dir, anno_dir, train_size, test_size):
    anno_file = os.path.join(anno_dir, 'umdfaces_batch3_ultraface.csv')
    dataset = []

    num_train_test_size = train_size + test_size

    count = 0
    with open(anno_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)
        for row in readCSV:

            per_data_info = []
            img_name = row[1]
            per_data_info.append(img_name)
            face_x = row[4]
            per_data_info.append(face_x)
            face_y = row[5]
            per_data_info.append(face_y)
            face_width = row[6]
            per_data_info.append(face_width)
            face_height = row[7]
            per_data_info.append(face_height)
            count += 1
            dataset.append(per_data_info)
            if (count < num_train_test_size):
                continue
            break

    print(dataset)
    return dataset


def create_datasets_UMD(image_dir, anno_dir, save_dir, use_Color, patch_size, train_size, test_size):
    dataset = parse_UMD(image_dir, anno_dir, train_size, test_size)

    # create directories for saving cropped datasets if necessary
    make_dir_if_not_exist(save_dir)
    dataset_tag = 'color' if use_Color else 'gray'

    save_folder = os.path.join(save_dir, dataset_tag)
    make_dir_if_not_exist(save_folder)

    dataset_train = 'train'
    save_folder_train = os.path.join(save_folder, dataset_train)
    make_dir_if_not_exist(save_folder_train)

    dataset_test = 'test'
    save_folder_test = os.path.join(save_folder, dataset_test)
    make_dir_if_not_exist(save_folder_test)

    # create pos folder-train
    positive = 'pos'
    positive_folder_train = os.path.join(save_folder_train, positive)
    make_dir_if_not_exist(positive_folder_train)

    # create neg folder-train
    negative = 'neg'
    negative_folder_train = os.path.join(save_folder_train, negative)
    make_dir_if_not_exist(negative_folder_train)

    # create pos folder-test
    positive = 'pos'
    positive_folder_test = os.path.join(save_folder_test, positive)
    make_dir_if_not_exist(positive_folder_test)

    # create neg folder-test
    negative = 'neg'
    negative_folder_test = os.path.join(save_folder_test, negative)
    make_dir_if_not_exist(negative_folder_test)

    count = 0
    dSize = patch_size

    img_tag = cv2.IMREAD_COLOR if use_Color else cv2.IMREAD_GRAYSCALE

    # train
    # crop positive
    for i in range(1, train_size):
        per_data_info = dataset[i]
        full_name = per_data_info[0]
        index_of_slash = full_name.find('/')
        img_name = full_name[index_of_slash + 1:]
        folder_name = img_name[:index_of_slash]
        X = int(float(per_data_info[1]))
        Y = int(float(per_data_info[2]))
        W = int(float(per_data_info[3]))
        H = int(float(per_data_info[4]))

        # find the image in the image_dir
        folder_path = os.path.join(image_dir, folder_name)

        # if use_Color is True
        if use_Color:

            for filename in glob.glob(folder_path + '/*.jpg'):

                length = len(img_name)
                filename_short = filename[len(filename) - length:]

                if (filename_short == img_name):
                    img = cv2.imread(folder_path + '/' + img_name, img_tag)
                    cv2.rectangle(img, (X, Y), (X + H, Y + H), (255, 255, 255))

                    # crop pos
                    cropped_image_pos = img[Y:Y + H, X:X + W]

                    # crop neg
                    cropped_image_neg = img[0:dSize, 0:dSize]

                    # save pos
                    resized_pos = cv2.resize(cropped_image_pos, (dSize, dSize))
                    cv2.imwrite(positive_folder_train + '/' + img_name, resized_pos)

                    # save neg
                    resized_neg = cv2.resize(cropped_image_neg, (dSize, dSize))
                    cv2.imwrite(negative_folder_train + '/' + img_name, resized_neg)

        # if use_Color is False
        else:

            for filename in glob.glob(folder_path + '/*.jpg'):

                length = len(img_name)
                filename_short = filename[len(filename) - length:]

                if (filename_short == img_name):
                    img = cv2.imread(folder_path + '/' + img_name, img_tag)
                    cv2.rectangle(img, (X, Y), (X + H, Y + H), (255, 255, 255))

                    # crop pos
                    cropped_image_pos = img[Y:Y + H, X:X + W]

                    # crop neg
                    cropped_image_neg = img[0:dSize, 0:dSize]

                    # save pos
                    resized_pos = cv2.resize(cropped_image_pos, (dSize, dSize))
                    cv2.imwrite(positive_folder_train + '/' + img_name, resized_pos)

                    # save neg
                    resized_neg = cv2.resize(cropped_image_neg, (dSize, dSize))
                    cv2.imwrite(negative_folder_train + '/' + img_name, resized_neg)

    # **************************************************************************

    for i in range(train_size, train_size + test_size):
        per_data_info = dataset[i]
        full_name = per_data_info[0]
        index_of_slash = full_name.find('/')
        img_name = full_name[index_of_slash + 1:]
        folder_name = img_name[:index_of_slash]
        X = int(float(per_data_info[1]))
        Y = int(float(per_data_info[2]))
        W = int(float(per_data_info[3]))
        H = int(float(per_data_info[4]))

        # find the image in the image_dir
        folder_path = os.path.join(image_dir, folder_name)
        print(folder_path)

        # if use_Color is True
        if use_Color:

            for filename in glob.glob(folder_path + '/*.jpg'):

                length = len(img_name)
                filename_short = filename[len(filename) - length:]

                if (filename_short == img_name):
                    img = cv2.imread(folder_path + '/' + img_name, img_tag)
                    cv2.rectangle(img, (X, Y), (X + H, Y + H), (255, 255, 255))

                    # crop pos
                    cropped_image_pos = img[Y:Y + H, X:X + W]

                    # crop neg
                    cropped_image_neg = img[0:dSize, 0:dSize]

                    # save pos
                    resized_pos = cv2.resize(cropped_image_pos, (dSize, dSize))
                    cv2.imwrite(positive_folder_test + '/' + img_name, resized_pos)

                    # save neg
                    resized_neg = cv2.resize(cropped_image_neg, (dSize, dSize))
                    cv2.imwrite(negative_folder_test + '/' + img_name, resized_neg)

        # if use_Color is False
        else:
            print('use color is false')
            for filename in glob.glob(folder_path + '/*.jpg'):

                length = len(img_name)
                filename_short = filename[len(filename) - length:]
                print(filename_short)

                if (filename_short == img_name):
                    img = cv2.imread(folder_path + '/' + img_name, img_tag)
                    cv2.rectangle(img, (X, Y), (X + H, Y + H), (255, 255, 255))

                    # crop pos
                    cropped_image_pos = img[Y:Y + H, X:X + W]

                    # crop neg
                    cropped_image_neg = img[Y:Y + H, X:X + W]

                    # save pos
                    resized = cv2.resize(cropped_image_pos, (dSize, dSize))
                    cv2.imwrite(positive_folder_test + '/' + img_name, resized)

                    # save neg
                    resized_neg = cv2.resize(cropped_image_neg, (dSize, dSize))
                    cv2.imwrite(negative_folder_test + '/' + img_name, resized_neg)


if __name__ == '__main__':
    create_datasets_UMD('face_data/original_pics', 'face_data/annotation', '/Users/nev/PycharmProjects/257-partA/face_data/cache', False, 60, 10, 10)