import skvideo.io
import cv2
import os
import csv
import glob
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
    def get_random_text_settings():
        """
        return random overlay configuration
        :return: dict()
        """
        overlay = dict()
        overlay['font'] = choice(FONT_SET)
        overlay['font_color'] = choice(COLOR_SET)
        overlay['font_scale'] = round(uniform(1, 1.5), 1)
        overlay['line_thickness'] = 2
        return overlay

    @staticmethod
    def read_file(file_name):
        """
        Generator read file and yield each line
        """
        with open(file_name) as fread:
            for line in fread:
                yield line

    @staticmethod
    def compute_text_coordinates(image, text,  overlay):
        """
        Function for computing coordinates of overlay by overlay parameters such
        text, font, font_scale, line_thickness
        :param image: numpy.ndarray`
        :param overlay: dict
        :return: (int, int) -> text_x_coordinate, text_y_coordinate
        """
        text_size = cv2.getTextSize(text,
                                    overlay['font'],
                                    overlay['font_scale'],
                                    overlay['line_thickness'])
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        image_width = image.shape[1]
        image_height = image.shape[0]
        text_x_coordinate = randint(10, int(image_width - text_width))
        text_y_coordinate = randint(int(text_height + 5), int(image_height - text_height))
        return text_x_coordinate, text_y_coordinate, text_width, text_height

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
                    # set random settings for overlay
                    text_settings = self.get_random_text_settings()
                    overlay = dict()
                    overlay['time_code'] = choice(self.TEXT['time_code_lines'])
                    overlay['date_lines'] = choice(self.TEXT['date_lines'])
                    overlay['speed_lines'] = choice(self.TEXT['speed_lines'])
                    overlay['camera_lines'] = choice(self.TEXT['camera_lines'])

                    # take frame from video
                    success, image = vidcap.read()
                    if image_index == self.images_per_video:
                        success = False
                    image = self.rotate_image(image, 360 - rotate_angle)


                    # Compute for each text - size of the text
                    # using sizes compute available amount of points for drawing on image
                    # draw txts as a pie
                    # using first x,y and concated heights and max(width) write bounding boxes
                    i = 0
                    for key in overlay:
                        text_x_coordinate, text_y_coordinate, text_width, text_height = self.compute_text_coordinates(
                            image, overlay[key], text_settings)
                        gap = text_height + 5
                        y = int((image.shape[0] + text_height) / 2) + i * gap
                        x = 10  # for center alignment => int((img.shape[1] - textsize[0]) / 2)
                        cv2.putText(image, overlay[key], (text_x_coordinate, y),
                                    text_settings['font'],
                                    text_settings['font_scale'],
                                    (255, 255, 255),
                                    text_settings['line_thickness'],
                                    lineType=cv2.LINE_AA)
                        i += 1
                    cv2.imshow("Result Image", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # Rotate image if we've got .mov file


                    # Compute bounding box coordinates for neural network education
                    x0 = text_x_coordinate - self.overlay_padding
                    x1 = text_x_coordinate + text_width + self.overlay_padding
                    y0 = text_y_coordinate - text_height - self.overlay_padding
                    y1 = text_y_coordinate + self.overlay_padding
                    # --------------------------------------------------------------------
                    img_file_name = video_file_name + "_sample%d.jpg" % image_index
                    cv2.imwrite(os.path.join(self.video_folder, img_file_name), image)

                    # Set class marks
                    image_index += 1
                    if font_color == (0, 0, 0):
                        class_name = 'black'
                        class_target = 2
                    if font_color == (230, 230, 230):
                        class_name = 'white'
                        class_target = 0

                    self.OUTPUT.append((os.path.join(images_dir, img_file_name),
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
                            break;
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
            file_index += 1

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
