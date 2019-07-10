import cv2
import os
import csv
from matplotlib import pyplot
from random import randint, choice
TEXT_FOLDER = 'recognition_dataset'
TRAIN_OUTPUT_FILE = "train.csv"
VALIDATION_OUTPUT_FILE = "validation.csv"

SPLIT_RATIO = 0.8


def generate_images(images_dir = None, video_sample_path='1st.mp4'):
    lines = None
    with open(os.path.join(TEXT_FOLDER, 'wordlist_mono_clean.txt'), 'r') as file:
        lines = file.readlines()
    output = []
    count = 0
    vidcap = cv2.VideoCapture(video_sample_path)
    vidcap.set(cv2.CAP_PROP_FPS, 1)
    success = True
    while success:
        # Set characteristics of text
        font = cv2.FONT_ITALIC
        font_scale = 2
        font_color = randint(70, 230)
        # font_color = 80
        line_type = 3

        # Take a frame from video recording
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500 + 6000))  # added this line
        success, image = vidcap.read()
        if not success:
            break

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get random timecode from list of codes
        text = choice(lines)[0:8]
        print(text)
        # print(text[7])
        # get text size
        text_size = cv2.getTextSize(text, font, font_scale, line_type)
        print(text_size[0][0])
        text_width = text_size[0][0]
        text_height = text_size[0][1]

        image_width = image.shape[1]
        image_height = image.shape[0]

        text_x_coordinate = randint(10, int(image_width - text_width))
        text_y_coordinate = randint(int(text_height + 5),  int(image_height - text_height))

        cv2.putText(image,
                    text,
                    (text_x_coordinate, text_y_coordinate),
                    font,
                    font_scale,
                    font_color,
                    line_type)

        x0 = text_x_coordinate - 7
        x1 = text_x_coordinate + text_width + 7

        y0 = text_y_coordinate - text_height - 7
        y1 = text_y_coordinate + 7
        # cv2.rectangle(image, (x0, y0), (x1, y1), 0, 3)
        # pyplot.imshow(image, cmap="gray")
        # pyplot.show()

        img_file_name = "_sample%d.jpg" % count
        cv2.imwrite(os.path.join(dirName, img_file_name), image)

        count += 1
        if font_color < 80:
            class_name = 'black'
            class_target = 2
        if font_color >= 80 & font_color <= 160:
            class_name = 'gray'
            class_target = 1
        if font_color > 160:
            class_name = 'white'
            class_target = 0

        output.append((os.path.join(dirName, img_file_name),
                       image_height, image_width,
                       x0, y0, x1, y1,
                       class_name, class_target))
        # print(output)
        # if count > 10:
        #     success = False


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

    print("class {}: {} images".format(output[j - 1][-2], i))
    lengths.append(i)

    with open(TRAIN_OUTPUT_FILE, "w") as train, open(VALIDATION_OUTPUT_FILE, "w") as validate:
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
    dirName = "det_set"
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Dataset directory:", dirName, " Created ")
    else:
        print("Dataset directory:", dirName, " already exists")

    generate_images(images_dir=dirName, video_sample_path='videos/sample.mp4')

