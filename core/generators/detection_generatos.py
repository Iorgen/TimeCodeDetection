# TODO create individual requirements file for gpu support
# TODO if something go wrong
import csv
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import *
from keras.utils import Sequence


class TimeCodeImageGenerator(Sequence):

    def __init__(self, csv_file, rnd_rescale=False, rnd_multiply=False, rnd_color=False,
                 rnd_crop=False, rnd_flip=False, debug=False, batch_size=32, image_size=448, grid_size=14):
        self.boxes = []
        self.rnd_rescale = rnd_rescale
        self.rnd_multiply = rnd_multiply
        self.rnd_color = rnd_color
        self.rnd_crop = rnd_crop
        self.rnd_flip = rnd_flip
        self.debug = debug
        self.batch_size = batch_size
        self.image_size = image_size
        self.grid_size = grid_size

        with open(csv_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)
                # x0, y0  top-left corner and x1,y1 bottom-right corner
                path, image_height, image_width, x0, y0, x1, y1, _, _ = row
                self.boxes.append((path, x0, y0, x1, y1))

    def __len__(self):
        return math.ceil(len(self.boxes) / self.batch_size)

    def __getitem__(self, idx):
        boxes = self.boxes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = np.zeros((len(boxes), self.image_size, self.image_size, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(boxes), self.grid_size, self.grid_size, 5), dtype=np.float32)
        for i, row in enumerate(boxes):
            path, x0, y0, x1, y1 = row

            with Image.open(os.path.join('train', path)) as img:
                if self.rnd_rescale:
                    old_width = img.width
                    old_height = img.height

                    rescale = np.random.uniform(low=0.6, high=1.4)
                    new_width = int(old_width * rescale)
                    new_height = int(old_height * rescale)

                    img = img.resize((new_width, new_height))

                    x0 *= new_width / old_width
                    y0 *= new_height / old_height
                    x1 *= new_width / old_width
                    y1 *= new_height / old_height

                if self.rnd_crop:
                    start_x = np.random.randint(0, high=np.floor(0.15 * img.width))
                    stop_x = img.width - np.random.randint(0, high=np.floor(0.15 * img.width))
                    start_y = np.random.randint(0, high=np.floor(0.15 * img.height))
                    stop_y = img.height - np.random.randint(0, high=np.floor(0.15 * img.height))

                    img = img.crop((start_x, start_y, stop_x, stop_y))

                    x0 = max(x0 - start_x, 0)
                    y0 = max(y0 - start_y, 0)
                    x1 = min(x1 - start_x, img.width)
                    y1 = min(y1 - start_y, img.height)

                    # if np.abs(x1 - x0) < 5 or np.abs(y1 - y0) < 5:
                    #     print("\nWarning: cropped too much (obj width {}, obj height {}, img width {}, img height {})\n".format(x1 - x0, y1 - y0, img.width, img.height))

                if self.rnd_flip:
                    elem = np.random.choice([0, 90, 180, 270, 1423, 1234])
                    if elem % 10 == 0:
                        x = x0 - img.width / 2
                        y = y0 - img.height / 2

                        x0 = img.width / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
                        y0 = img.height / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

                        x = x1 - img.width / 2
                        y = y1 - img.height / 2

                        x1 = img.width / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
                        y1 = img.height / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

                        img = img.rotate(-elem)
                    else:
                        if elem == 1423:
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            y0 = img.height - y0
                            y1 = img.height - y1

                        elif elem == 1234:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            x0 = img.width - x0
                            x1 = img.width - x1

                image_width = img.width
                image_height = img.height

                tmp = x0
                x0 = min(x0, x1)
                x1 = max(tmp, x1)

                tmp = y0
                y0 = min(y0, y1)
                y1 = max(tmp, y1)

                x0 = max(x0, 0)
                y0 = max(y0, 0)

                y0 = min(y0, image_height)
                x0 = min(x0, image_width)
                y1 = min(y1, image_height)
                x1 = min(x1, image_width)

                if self.rnd_color:
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(np.random.uniform(low=0.5, high=1.5))

                    enhancer2 = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(np.random.uniform(low=0.7, high=1.3))

                img = img.resize((self.image_size, self.image_size))
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32)

                if self.rnd_multiply:
                    img[...,0] = np.floor(np.clip(img[...,0] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))
                    img[...,1] = np.floor(np.clip(img[...,1] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))
                    img[...,2] = np.floor(np.clip(img[...,2] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))

                batch_images[i] = preprocess_input(img.copy())

            x_c = ((self.grid_size - 1) / image_width) * (x0 + (x1 - x0) / 2)
            y_c = ((self.grid_size - 1) / image_height) * (y0 + (y1 - y0) / 2)

            floor_y = math.floor(y_c)
            floor_x = math.floor(x_c)

            batch_boxes[i, floor_y, floor_x, 0] = (y1 - y0) / image_height
            batch_boxes[i, floor_y, floor_x, 1] = (x1 - x0) / image_width
            batch_boxes[i, floor_y, floor_x, 2] = y_c - floor_y
            batch_boxes[i, floor_y, floor_x, 3] = x_c - floor_x
            batch_boxes[i, floor_y, floor_x, 4] = 1

            if self.debug:
                changed = img.astype(np.uint8)
                if not os.path.exists("__debug__"):
                    os.makedirs("__debug__")

                changed = Image.fromarray(changed)

                x_c = (floor_x + batch_boxes[i, floor_y, floor_x, 3]) / (self.grid_size - 1)
                y_c = (floor_y + batch_boxes[i, floor_y, floor_x, 2]) / (self.grid_size - 1)

                y0 = self.image_size * (y_c - batch_boxes[i, floor_y, floor_x, 0] / 2)
                x0 = self.image_size * (x_c - batch_boxes[i, floor_y, floor_x, 1] / 2)
                y1 = y0 + self.image_size * batch_boxes[i, floor_y, floor_x, 0]
                x1 = x0 + self.image_size * batch_boxes[i, floor_y, floor_x, 1]

                draw = ImageDraw.Draw(changed)
                draw.rectangle(((x0, y0), (x1, y1)), outline="green")

                changed.save(os.path.join("__debug__", os.path.basename(path)))
        #
        # for idx, image in enumerate(batch_images[:5]):
        #     pure_image = image
        #     box_image = cv2.rectangle(pure_image, (50, 50), (50 + 10, 50 + 20), (255, 255, 00), 2)
        #     pyplot.imshow(pure_image)
        #     pyplot.show()
        #     pyplot.imshow(box_image)
        #     pyplot.show()
        return batch_images, batch_boxes