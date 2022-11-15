import numpy as np
import os
import cv2
import sys


def initialize(image):
    # these values were determined via manual inspection
    top_left = (19, 50)
    bottom_right = (67, 92)
    image = draw_box(image, top_left, bottom_right)
    return image, top_left, bottom_right


# function to draw a pink box bounded by tuples top_left and bottom_right
def draw_box(image, top_left, bottom_right):

    # check the boundaries of the top left value
    if not (image.shape[0] > top_left[0] > -1 and image.shape[1] > top_left[1] > -1):
        print("ERROR: invalid top left location provided!")
        exit(1)

    # check the boundaries of the bottom right pixel
    if not(image.shape[0] > bottom_right[0] > -1 and image.shape[1] > bottom_right[1] > -1):
        print("ERROR: invalid bottom right location provided!")
        exit(1)

    # draw the left and right edges
    for i in range(top_left[0], bottom_right[0] + 1):
        image[i, top_left[1]] = [255, 0, 255]
        image[i, bottom_right[1]] = [255, 0, 255]

    # draw the top and bottom edges
    for i in range(top_left[1], bottom_right[1] + 1):
        image[top_left[0], i] = [255, 0, 255]
        image[bottom_right[0], i] = [255, 0, 255]

    return image


def ssd(image, next_image, top_left, bottom_right):
    # search with a buffer of n pixels around the current image, finding the location in the next image that most closely matches
    buff = 6
    ssd_arr = np.zeros((2*buff+1, 2*buff+1))
    for vert_shift in range(-buff, buff+1):
        for horiz_shift in range(-buff, buff+1):
            if (top_left[0] + vert_shift < 0) or (bottom_right[0] + 1 + vert_shift >= image.shape[0]) or (top_left[1] + horiz_shift < 0) or (bottom_right[1] + 1 + horiz_shift >= image.shape[1]):
                ssd_arr[vert_shift + buff, horiz_shift + buff] = sys.maxsize
                continue
            ssd_arr[vert_shift + buff, horiz_shift + buff] = np.sum((image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] - next_image[top_left[0] + vert_shift:bottom_right[0]+1+vert_shift, top_left[1]+horiz_shift:bottom_right[1]+1+horiz_shift]) ** 2)
    sorted_by_ssd = np.argsort(ssd_arr.flatten())
    # the new box is described by the minimum result in the ssd
    # the row is going to be int(arg / width)
    # the column is going to be (arg % width)
    new_TL_row = top_left[0] + int(sorted_by_ssd[0] / (2 * buff + 1)) - buff
    new_TL_col = top_left[1] + (sorted_by_ssd[0] % (2 * buff + 1)) - buff
    new_BR_row = new_TL_row + (bottom_right[0] - top_left[0])
    new_BR_col = new_TL_col + (bottom_right[1] - top_left[1])
    top_left = (new_TL_row, new_TL_col)
    bottom_right = (new_BR_row, new_BR_col)
    return top_left, bottom_right


def cross_correlation(image, top_left, bottom_right):
    return


def norm_cc(image, top_left, bottom_right):
    return


def main():
    folder = 'image_girl'
    frame_files = sorted(os.listdir(folder))

    # test initial box
    image = cv2.imread(folder + '/' + frame_files[0])
    image = np.array(image)
    image, top_left, bottom_right = initialize(image)
    cv2.imwrite('sse_boxed/' + frame_files[0], image)

    for frame_idx in range(1, len(frame_files) - 1):
        next_image = cv2.imread(folder + '/' + frame_files[frame_idx])
        top_left, bottom_right = ssd(image, next_image, top_left, bottom_right)
        boxed_image = draw_box(next_image, top_left, bottom_right)
        cv2.imwrite('sse_boxed/' + frame_files[frame_idx], boxed_image)
        image = next_image
    return


if __name__ == '__main__':
    main()
