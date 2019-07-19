from detection_train import *
import cv2
import glob
from matplotlib import pyplot
import numpy as np

WEIGHTS_FILE = "model-0.18.h5"
IMAGES = "detection_dataset/images/*jpg"
IMAGE_SIZE = 224
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2
MAX_OUTPUT_SIZE = 49


def main():
    count = 0
    model = create_model()
    model.load_weights(WEIGHTS_FILE)
    vidcap = cv2.VideoCapture('videos/1.mp4')
    vidcap.set(cv2.CAP_PROP_FPS, 1)
    success = True
    while success:
        # Set characteristics of text
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500 + 6000))  # added this line
        success, image = vidcap.read()
        unscaled = image
        # print(filename)
        img = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))

        feat_scaled = preprocess_input(np.array(img, dtype=np.float32))

        pred = np.squeeze(model.predict(feat_scaled[np.newaxis, :]))
        print(pred)
        height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]

        coords = np.arange(pred.shape[0] * pred.shape[1])
        y = (y_f + coords // pred.shape[0]) / (pred.shape[0] - 1)
        x = (x_f + coords % pred.shape[1]) / (pred.shape[1] - 1)

        boxes = np.stack([y, x, height, width, score], axis=-1)
        boxes = boxes[np.where(boxes[..., -1] >= SCORE_THRESHOLD)]

        selected_indices = tf.image.non_max_suppression(boxes[..., :-1], boxes[..., -1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
        selected_indices = tf.Session().run(selected_indices)

        for y_c, x_c, h, w, _ in boxes[selected_indices]:
            print(h, w)
            x0 = unscaled.shape[1] * (x_c - w / 2)
            y0 = unscaled.shape[0] * (y_c - h / 2)
            x1 = x0 + unscaled.shape[1] * w
            y1 = y0 + unscaled.shape[0] * h

            cv2.rectangle(unscaled, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)
            crop_img = unscaled[int(y0):int(y1), int(x0):int(x1)]
            pyplot.imshow(crop_img)
            pyplot.show()

        pyplot.imshow(image)
        pyplot.show()
        count += 1
        # cv2.imshow("image", unscaled)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # for filename in glob.glob(IMAGES):
    #
    #
    #
    #     unscaled = cv2.imread(filename)
    #     print(filename)
    #     img = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
    #
    #     feat_scaled = preprocess_input(np.array(img, dtype=np.float32))
    #
    #     pred = np.squeeze(model.predict(feat_scaled[np.newaxis,:]))
    #     print(pred)
    #     height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]
    #
    #     coords = np.arange(pred.shape[0] * pred.shape[1])
    #     y = (y_f + coords // pred.shape[0]) / (pred.shape[0] - 1)
    #     x = (x_f + coords % pred.shape[1]) / (pred.shape[1] - 1)
    #
    #     boxes = np.stack([y, x, height, width, score], axis=-1)
    #     boxes = boxes[np.where(boxes[...,-1] >= SCORE_THRESHOLD)]
    #
    #     selected_indices = tf.image.non_max_suppression(boxes[...,:-1], boxes[...,-1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
    #     selected_indices = tf.Session().run(selected_indices)
    #
    #     for y_c, x_c, h, w, _ in boxes[selected_indices]:
    #         print(h,w)
    #         x0 = unscaled.shape[1] * (x_c - w / 2)
    #         y0 = unscaled.shape[0] * (y_c - h / 2)
    #         x1 = x0 + unscaled.shape[1] * w
    #         y1 = y0 + unscaled.shape[0] * h
    #
    #         cv2.rectangle(unscaled, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)
    #
    #     cv2.imshow("image", unscaled)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()