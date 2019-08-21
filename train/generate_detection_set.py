import skvideo.io
import json
import numpy as np
import cv2
import os
import csv
import glob
import errno
from matplotlib import pyplot
from random import randint, choice, uniform
TEXT_FOLDER = 'recognition_dataset'
# DETECTION_FOLDER = os.path.join("train", "detection_dataset")
DETECTION_FOLDER = "detection_dataset"
TRAIN_OUTPUT_FILE = os.path.join("detection_dataset", "train.csv")
VALIDATION_OUTPUT_FILE = os.path.join("detection_dataset", "validation.csv")

SPLIT_RATIO = 0.8
# COLOR_SET = [(20, 20, 20), (160, 160, 160), (230, 230, 230)]
# Two color mode
COLOR_SET = [(0, 0, 0), (230, 230, 230)]
# TODO upload another fonts and generate text based on them
FONT_SET = [0, 1, 2, 3, 4, 5, 6, 7, 16]


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def generate_images(images_dir=None, video_folder='videos', debug=False):
    with open(os.path.join(TEXT_FOLDER, 'wordlist_mono_clean.txt'), 'r') as file:
        lines = file.readlines()
        output = []
    file_index = 0
    videos = glob.glob(os.path.join(DETECTION_FOLDER, video_folder, '*.mov'))
    # videos = glob.glob(os.path.join(DETECTION_FOLDER, video_folder, '*.mp4'))
    for file in videos:
        print(file_index)
        try:
            image_index = 0
            # vidcap = skvideo.io.vread(file)
            metadata = skvideo.io.ffprobe(file)
            # metadata = json.dumps(metadata["video"], indent=4)
            rotate_angle = metadata['video']['tag'][0]['@value']
            print(rotate_angle)

            try:
                rotate_angle = int(rotate_angle)
            except Exception as e:
                rotate_angle = 360
                pass
            vidcap = cv2.VideoCapture(file)
            # vidcap.set(cv2.CAP_PROP_FPS, 1)
            success = True
            while success:
                # Set characteristics of text
                font = choice(FONT_SET)
                font_color = choice(COLOR_SET)
                font_scale = round(uniform(1, 1.5), 1)
                line_thickness = 2

                # Take a frame from video recording
                vidcap.set(cv2.CAP_PROP_POS_MSEC, (image_index * 1000))  # added this line
                success, image = vidcap.read()
                if image_index==10:
                    success=False
                # if not success:
                #     break

                image = rotate_image(image, 360 - rotate_angle)
                # Get random timecode from list of codes
                text = choice(lines)[0:8]

                # get text size and compute coordinates
                text_size = cv2.getTextSize(text, font, font_scale, line_thickness)
                text_width = text_size[0][0]
                text_height = text_size[0][1]
                image_width = image.shape[1]
                image_height = image.shape[0]
                text_x_coordinate = randint(10, int(image_width - text_width))
                text_y_coordinate = randint(int(text_height + 5), int(image_height - text_height))

                # paint text on image
                cv2.putText(image,
                            text,
                            (text_x_coordinate, text_y_coordinate),
                            font,
                            font_scale,
                            font_color,
                            line_thickness)

                # Compute bounding box coordinates for neural network education
                x0 = text_x_coordinate - 15
                x1 = text_x_coordinate + text_width + 15
                y0 = text_y_coordinate - text_height - 15
                y1 = text_y_coordinate + 15

                img_file_name = str(file_index) + "_sample%d.jpg" % image_index
                cv2.imwrite(os.path.join(images_dir, img_file_name), image)

                image_index += 1
                if font_color == (0, 0, 0):
                    class_name = 'black'
                    class_target = 2
                if font_color == (230, 230, 230):
                    class_name = 'white'
                    class_target = 0

                output.append((os.path.join(images_dir, img_file_name),
                               image_height, image_width,
                               x0, y0, x1, y1,
                               class_name, class_target))

                if debug:
                    print("font:", font)
                    print("font-color", font_color)
                    print("font scale", font_scale)
                    print("line thickness", line_thickness)
                    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)
                    pyplot.imshow(image)
                    pyplot.show()
                    if image_index > 4:
                        success = False
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
        file_index +=1

    # preserve percentage of samples for each class ("stratified")
    output.sort(key=lambda tup: tup[-1])

    lengths = []
    i = 0
    last = 0
    for j, row in enumerate(output):
        if last == row[-1]:
            i += 1
        else:
            print("class {}: {} images".format(output[j - 1][-2], i))
            lengths.append(i)
            i = 1
            last += 1

    # print("class {}: {} images".format(output[j - 1][-2], i))
    lengths.append(i)

    with open(TRAIN_OUTPUT_FILE, "w", newline='') as train, open(VALIDATION_OUTPUT_FILE, "w", newline='') as validate:
        writer = csv.writer(train, delimiter=",")
        writer2 = csv.writer(validate, delimiter=",")

        s = 0
        for c in lengths:
            for i in range(c):
                print("{}/{}".format(s + 1, sum(lengths)), end="\r")

                path, height, width, xmin, ymin, xmax, ymax, class_name, class_id = output[s]

                # if xmin >= xmax or ymin >= ymax or xmax > width or ymax > height or xmin < 0 or ymin < 0:
                #     print("Warning: {} contains invalid box. Skipped...".format(path))
                #     continue

                row = [path, height, width, xmin, ymin, xmax, ymax, class_name, class_id]
                if i <= c * SPLIT_RATIO:
                    writer.writerow(row)
                else:
                    writer2.writerow(row)

                s += 1

    print("\nDone!")


if __name__ == '__main__':
    # create images directory
    dirName = os.path.join("detection_dataset", "image_set")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Dataset directory:", dirName, " Created ")
    else:
        print("Dataset directory:", dirName, " already exists")

    generate_images(images_dir=dirName, video_folder="videos", debug=False)
    print("generate complete successful")


