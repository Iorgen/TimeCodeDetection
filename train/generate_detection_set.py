import skvideo.io
import cv2
import os
import csv
import glob
import numpy
import errno
import json
from matplotlib import pyplot
from random import randint, choice, uniform
TEXT_FOLDER = 'recognition_dataset'
DETECTION_FOLDER = "detection_dataset"
TRAIN_OUTPUT_FILE = os.path.join("detection_dataset", "train.csv")
VALIDATION_OUTPUT_FILE = os.path.join("detection_dataset", "validation.csv")
SPLIT_RATIO = 0.8
COLOR_SET = [(0, 0, 0), (230, 230, 230)]
FONT_SET = [0, 1, 2, 3, 4, 5, 6, 7, 16]


class OverLay:

    COLOR_SET = [(0, 0, 0), (230, 230, 230)]
    FONT_SET = [0, 1, 2, 3, 4, 5, 6, 7, 16]

    def __init__(self, time_code="", date="", speed="", camera=""):
        self.text = dict()
        self.text_width = dict()
        self.text_height = dict()
        self.width = 0
        self.height = 0
        # Set text characteristics
        self.font = choice(FONT_SET)
        self.font_scale = round(uniform(1, 1.5), 1)
        self.font_color = choice(COLOR_SET)
        self.line_thickness = 2
        # Save text values
        self.text['time_code'] = time_code
        self.text['date'] = date
        self.text['speed'] = speed
        self.text['camera'] = camera
        #

        for key in self.text:
            self.text_width[key] = cv2.getTextSize(self.text[key],
                                                   self.font, self.font_scale, self.line_thickness)[0][0]
            self.text_height[key] = cv2.getTextSize(self.text[key],
                                                    self.font, self.font_scale, self.line_thickness)[0][1]
            self.height += self.text_height[key]
        self.width = self.text_width[max(self.text_width, key=lambda key: self.text_width[key])]


class DetectionDatasetGenerator:
    TEXT_FOLDER = 'recognition_dataset'
    DETECTION_FOLDER = "detection_dataset"
    TRAIN_OUTPUT_FILE = os.path.join("detection_dataset", "train.csv")
    VALIDATION_OUTPUT_FILE = os.path.join("detection_dataset", "validation.csv")
    SPLIT_RATIO = 0.8
    TEXT = dict()
    COLOR_SET = [(0, 0, 0), (230, 230, 230)]
    FONT_SET = [0, 1, 2, 3, 4, 5, 6, 7, 16]
    OUTPUT = []

    def __init__(self, video_folder='videos', images_per_video=10, max_objects_on_image=3,
                 overlay_padding=15):
        self.images_per_video = images_per_video
        self.video_folder = video_folder
        self.max_objects_on_image = max_objects_on_image
        self.overlay_padding = overlay_padding
        with open(os.path.join('configuration', 'recognition.json'), 'r') as config:
            self.configuration = json.load(config)

        with open(os.path.join(TEXT_FOLDER, self.configuration['TIME_CODE_FILE']), 'r') as file:
            self.TEXT['time_code_lines'] = [x.rstrip("\n") for x in file.readlines()]
        with open(os.path.join(TEXT_FOLDER, self.configuration['DATE_FILE']), 'r') as file:
            self.TEXT['date_lines'] = [x.rstrip("\n") for x in file.readlines()]
        # with open(os.path.join(TEXT_FOLDER, self.configuration['DAY_WEEK_FILE']), 'r') as file:
        #     self.day_week_lines = file.readlines()
        with open(os.path.join(TEXT_FOLDER, self.configuration['SPEED_FILE']), 'r') as file:
            self.TEXT['speed_lines'] = [x.rstrip("\n") for x in file.readlines()]
        with open(os.path.join(TEXT_FOLDER, self.configuration['CAMERA_FILE']), 'r') as file:
            self.TEXT['camera_lines'] = [x.rstrip("\n") for x in file.readlines()]
        print(choice(list(self.TEXT.keys())))
        print(choice(self.TEXT[choice(list(self.TEXT.keys()))]))

    @staticmethod
    def video_files(path):
        """
        Generator read video and yield each video file
        :param path: string
        :return:
        """
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    @staticmethod
    def rotate_image(mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        :param mat: numpy.ndarray
        :param angle: float
        :return: numpy.ndarray
        """
        # image shape has 3 dimensions
        height, width = mat.shape[:2]
        # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        image_center = (width/2, height/2)

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

    @staticmethod
    def get_video_rotation(file_path):
        """
        Check if video include rotation angle as angel, not as text
        :param file_path: string
        :return: float
        """
        metadata = skvideo.io.ffprobe(file_path)
        try:
            rotate_angle = metadata['video']['tag'][0]['@value']
            rotate_angle = int(rotate_angle)
        except Exception as e:
            rotate_angle = 360
            pass
        return rotate_angle


    @staticmethod
    def read_file(file_name):
        """
        Generator read file and yield each line
        """
        with open(file_name) as fread:
            for line in fread:
                yield line

    @staticmethod
    def compute_overlay_coordinates(image, overlay):
        """
        Function for computing coordinates of overlay by overlay parameters such
        text, font, font_scale, line_thickness
        :param image: numpy.ndarray`
        :param overlay: class OverLay
        :return: (int, int)
        """

        # assert isinstance(overlay, OverLay)
        # assert isinstance(image, numpy.ndarray)
        image_width = image.shape[1]
        image_height = image.shape[0]
        overlay_x = randint(10, int(image_width - overlay.width))
        overlay_y = randint(int(overlay.height + 5), int(image_height - overlay.height))
        return overlay_x, overlay_y

    def generate_images(self, images_dir=None, debug=False):
        for video_file_name in self.video_files(os.path.join(self.DETECTION_FOLDER, self.video_folder)):
            file = os.path.join(
                self.DETECTION_FOLDER,
                self.video_folder,
                video_file_name)

            try:
                image_index = 0
                rotate_angle = self.get_video_rotation(file)
                vidcap = cv2.VideoCapture(file)
                success = True
                while success:
                    # take frame from video
                    success, image = vidcap.read()
                    if image_index == self.images_per_video:
                        success = False
                    # Rotate image if we've got .mov file
                    image = self.rotate_image(image, 360 - rotate_angle)
                    # set random settings for overlay and random text from dictionary
                    overlay = OverLay(choice(self.TEXT['time_code_lines']),
                                      choice(self.TEXT['date_lines']), 
                                      choice(self.TEXT['speed_lines']), 
                                      choice(self.TEXT['camera_lines']))

                    overlay_x, overlay_y = self.compute_overlay_coordinates(image, overlay)

                    # draw txts as a pie
                    i = 0
                    for key in overlay.text:
                        gap = overlay.text_height[key] + 5
                        y = overlay_y + i * gap
                        x = overlay_x  # for center alignment => int((img.shape[1] - textsize[0]) / 2)
                        cv2.putText(image, overlay.text[key], (x, y),
                                    overlay.font,
                                    overlay.font_scale,
                                    overlay.font_color,
                                    overlay.line_thickness,
                                    lineType=cv2.LINE_AA)
                        i += 1

                    # using first x,y and concated heights and max(width) write bounding boxes
                    x0 = overlay_x - self.overlay_padding
                    x1 = overlay_x + overlay.width + self.overlay_padding
                    y0 = overlay_y - overlay.text_height['time_code']
                    y1 = overlay_y + overlay.height
                    print(x0, x1, y0, y1)
                    # --------------------------------------------------------------------
                    img_file_name = video_file_name + "_sample%d.jpg" % image_index
                    cv2.imwrite(os.path.join(images_dir, img_file_name), image)

                    # Set class marks
                    image_index += 1
                    if overlay.font_color == (0, 0, 0):
                        class_name = 'black'
                        class_target = 2
                    if overlay.font_color == (230, 230, 230):
                        class_name = 'white'
                        class_target = 0

                    self.OUTPUT.append((os.path.join(images_dir, img_file_name),
                                   image.shape[0], image.shape[1],
                                   x0, y0, x1, y1,
                                   class_name, class_target))

                    if debug:
                        print("font:", overlay.font)
                        print("font-color", overlay.font_color)
                        print("font scale", overlay.font_scale)
                        print("line thickness", overlay.line_thickness)
                        # cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 3)
                        cv2.rectangle(image, (int(x0), int(y0)), (int(x0+ 10), int(y0+ 10)),
                                      (0, 255, 255), 5)
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + 10), int(y1 + 10)),
                                      (0, 255, 0), 5)

                        pyplot.imshow(image)
                        pyplot.show()
                        if image_index > 4:
                            success = False
                            break
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

        # preserve percentage of samples for each class ("stratified")
        self.OUTPUT.sort(key=lambda tup: tup[-1])
        lengths = []
        i = 0
        last = 0
        for j, row in enumerate(self.OUTPUT):
            if last == row[-1]:
                i += 1
            else:
                print("class {}: {} images".format(self.OUTPUT[j - 1][-2], i))
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

                    path, height, width, xmin, ymin, xmax, ymax, class_name, class_id = self.OUTPUT[s]

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

    generator = DetectionDatasetGenerator(video_folder="videos")
    generator.generate_images(images_dir=dirName, debug=True)
    print("generate complete successful")

# Go to frame
# vidcap.set(cv2.CAP_PROP_POS_MSEC, (image_index * 1000))  # added this line
