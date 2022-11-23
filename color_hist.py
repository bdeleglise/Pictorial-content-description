import cv2
import os
import matplotlib.pyplot as plt
import time
import numpy as np

import utils
from constant import DEBUG, BANQUE_IMAGES_PATH, HIST_RANGE, HIST_BIN, REQUEST_IMAGES_PATH, TOP_VOTING, THRESHOLD_DIST, \
    RESULT_CATEGORIES, RESULT_COLOR_HIST_FILE_PATH


def get_image_and_color_hists(file, path):
    image = cv2.imread(path + "/" + file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if DEBUG:
        plt.imshow(image)
        plt.show()

        height = image.shape[0]
        width = image.shape[1]

        print('Image Height       : ', height)
        print('Image Width        : ', width)

    red_hist = cv2.calcHist([image], [0], None, [HIST_BIN], [0, HIST_RANGE])
    green_hist = cv2.calcHist([image], [1], None, [HIST_BIN], [0, HIST_RANGE])
    blue_hist = cv2.calcHist([image], [2], None, [HIST_BIN], [0, HIST_RANGE])

    if DEBUG:
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [HIST_BIN], [0, HIST_RANGE])
            plt.plot(histr, color=col)
            plt.xlim([0, HIST_BIN])
        plt.show()

    return image, red_hist, green_hist, blue_hist


# INSPIRED from : https://medium.com/mlearning-ai/how-to-plot-color-channels-histogram-of-an-image-in-python-using-opencv-40022032e127
def create_reference_hist():
    files = os.listdir(BANQUE_IMAGES_PATH)
    color_hists = []

    for file in files:
        image, red_hist, green_hist, blue_hist = get_image_and_color_hists(file, BANQUE_IMAGES_PATH)

        hists_image = [red_hist, green_hist, blue_hist]
        color_hists.append({
            "image": file,
            "hists": hists_image
        })

    return color_hists


def image_recognition(color_hists_ref, time_create_hist_ref):
    files = os.listdir(REQUEST_IMAGES_PATH)
    test_pass = 0

    for file in files:
        start = time.time()
        image, red_hist, green_hist, blue_hist = get_image_and_color_hists(file, REQUEST_IMAGES_PATH)

        red_hist_normalize = red_hist / np.linalg.norm(red_hist)
        green_hist_normalize = green_hist / np.linalg.norm(green_hist)
        blue_hist_normalize = blue_hist / np.linalg.norm(blue_hist)

        distances = {}
        for reference in color_hists_ref:
            hists_ref = reference['hists']
            red_hist_ref = hists_ref[0]
            green_hist_ref = hists_ref[1]
            blue_hist_ref = hists_ref[2]

            red_hist_ref_normalize = red_hist_ref / np.linalg.norm(red_hist_ref)
            green_hist_ref_normalize = green_hist_ref / np.linalg.norm(green_hist_ref)
            blue_hist_ref_normalize = blue_hist_ref / np.linalg.norm(blue_hist_ref)

            res_vect_concat = np.concatenate(
                ((red_hist_normalize - red_hist_ref_normalize),
                 (green_hist_normalize - green_hist_ref_normalize),
                 (blue_hist_normalize - blue_hist_ref_normalize)),
                axis = None
            )

            if DEBUG:
                print(reference['image'])
                plt.subplot(3, 1, 1)
                plt.plot(np.concatenate((red_hist_normalize, green_hist_normalize, blue_hist_normalize), axis = None ))
                plt.xlim([0, HIST_BIN*3])
                plt.subplot(3, 1, 2)
                plt.plot(np.concatenate((red_hist_ref_normalize, green_hist_ref_normalize, blue_hist_ref_normalize), axis = None ))
                plt.xlim([0, HIST_BIN*3])
                plt.subplot(3, 1, 3)
                plt.plot(res_vect_concat)
                plt.xlim([0, HIST_BIN*3])

                plt.tight_layout()
                plt.show()

            dist = np.linalg.norm(res_vect_concat)

            distances[reference['image']] = dist

        distances = dict(sorted(distances.items(), key=lambda item: item[1]))

        if DEBUG:
            print(distances)

        top = {}
        for i in range(0, TOP_VOTING):
            file_ref = list(distances.keys())[i]
            if distances[file_ref] <= THRESHOLD_DIST:
                top[file_ref] = distances[file_ref]

        if DEBUG:
            print(top)

        elements = {}
        for i in range(0, len(top)):
            element = list(top.keys())[i]
            pos = element.find('_')
            element = element[0:pos]

            if element not in elements.keys():
                elements[element] = 0

            elements[element] = elements[element] + 1

        winner = None

        if len(top) != 0:
            elements = dict(sorted(elements.items(), key=lambda item: item[1], reverse=True))
            keys = list(elements.keys())
            winner = keys[0]
            if len(elements) > 1 and elements[keys[0]] == elements[keys[1]]:
                keys = list(top.keys())
                winner = keys[0]
                pos = winner.find('_')
                winner = winner[0:pos]

        if DEBUG:
            print(winner)

        test_ok = (winner == RESULT_CATEGORIES[file])
        end = time.time()
        exec_time = end - start

        utils.save_result(file, time_create_hist_ref, exec_time, distances, top, winner, test_ok,
                          RESULT_COLOR_HIST_FILE_PATH)

        if test_ok:
            test_pass = test_pass + 1

    precision = test_pass/len(files)
    f = open(RESULT_COLOR_HIST_FILE_PATH, "a")
    f.write("\nPrécision de l'algorithme ;" + str(precision) + "\n")