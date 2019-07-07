import cv2
import glob
from matplotlib import pyplot
from random import randint, randrange
# ../static/image/_frame0.jpg'
# recognition_train_data/*jpg


def paint_text(input_text, width, height, sample_image_path='../static/image/_frame0.jpg'):
    '''
    :param input_text:  text to be placed on the images
    :param width: ended width of generated images
    :param height: ended height of generated images
    :param sample_image_path: path to images samples on which the overlay occurs (default: "datasets/train_data/*jpg")
    :return: np.array()
    '''

    font = cv2.FONT_ITALIC
    # choose up or down
    # choose
    bottom_left_corner = (200
                          + randint(-4, 4), 200 + randint(-5, 5))  # That parameter should be random set
    font_scale = 0.7
    font_color = 230
    line_type = 2

    # imgs = glob(sample_image_path)
    # img_num = randrange(len(imgs))
    img = cv2.imread(sample_image_path, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.putText(img,
                input_text,
                bottom_left_corner,
                font,
                font_scale,
                font_color,
                line_type)

    pyplot.imshow(img, cmap=pyplot.cm.gray)
    pyplot.show()
    # img = cv2.resize(img, (width, height))
    cv2.imwrite('image.jpg', img)
    return img.reshape(1, height, width)


if __name__ == '__main__':
    h = 35
    w = 135
    generate_image = paint_text('22:22:22', w, h)
    # print(generate_image.shape)
    # generate_image = generate_image.reshape(generate_image.shape[1], generate_image.shape[2])

