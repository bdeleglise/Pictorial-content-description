import cv2
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.signal as sig
from skimage.feature import hog

from skimage import feature
from skimage import exposure
from skimage.transform import resize, pyramid_gaussian

import utils
from constant import DEBUG, BANQUE_IMAGES_PATH, REQUEST_IMAGES_PATH, TOP_VOTING, THRESHOLD_DIST, RESULT_CATEGORIES, \
    RESULT_HOG_FILE_PATH, SOBEL_FILTER, CELL_SIZE, BLOCK_SIZE, ABS_ANGLE, WIDTH, HEIGHT


"""
Allows to normalize an hog block 
Source : https://github.com/scikit-image/scikit-image/blob/fe96435877f40581d678b10fde650c6e1899354a/skimage/feature/_hog.py
"""
def hog_normalize_block(block, eps=1e-5):
    return block / np.sqrt(np.sum(block ** 2) + eps ** 2)


"""
Allows to make the convolution of an image with Sobel filter for each pixels of the image in order to get the gradient
Source : https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/
"""
def compute_gradient(image: np.ndarray):
    """
    Compute gradient of an image by rows and columns
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    G_x = sig.convolve2d(image, kernel_x, mode='same')
    G_y = sig.convolve2d(image, kernel_y, mode='same')

    return G_x, G_y


"""
Allows for a cell of a block to compute the histogram of oriented gradient value
Source : https://github.com/scikit-image/scikit-image/blob/fe96435877f40581d678b10fde650c6e1899354a/skimage/feature/_hoghistogram.pyx
"""
def cell_hog(magnitude,
             orientation,
             orientation_start, orientation_end,
             cell_columns, cell_rows,
             column_index, row_index,
             size_columns, size_rows,
             range_rows_start, range_rows_stop,
             range_columns_start, range_columns_stop):
    total = 0.
    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = row_index + cell_row
        if (cell_row_index < 0 or cell_row_index >= size_rows):
            continue

        for cell_column in range(range_columns_start, range_columns_stop):
            cell_column_index = column_index + cell_column
            if (cell_column_index < 0 or cell_column_index >= size_columns
                    or orientation[cell_row_index, cell_column_index]
                    >= orientation_start
                    or orientation[cell_row_index, cell_column_index]
                    < orientation_end):
                continue

            # compute the weighting to know which proportion of the angle we need to assign to the current interval
            # of the hist
            total += magnitude[cell_row_index, cell_column_index]

    return total / (cell_rows * cell_columns)


"""
Allows for each block of an image to compute HOGs of each cell and concatenate it
Source : https://github.com/scikit-image/scikit-image/blob/fe96435877f40581d678b10fde650c6e1899354a/skimage/feature/_hoghistogram.pyx
"""
def hog_histograms(gradient_columns,
                   gradient_rows,
                   cell_columns, cell_rows,
                   size_columns, size_rows,
                   number_of_cells_columns, number_of_cells_rows,
                   number_of_orientations,
                   orientation_histogram):

    # Magnitude of gradient
    magnitude = np.hypot(gradient_columns,
                         gradient_rows)

    # Angle of gradient
    orientation = \
        np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % (180 * ABS_ANGLE)

    r_0 = cell_rows / 2
    c_0 = cell_columns / 2
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    range_rows_stop = (cell_rows + 1) / 2
    range_rows_start = -(cell_rows / 2)
    range_columns_stop = (cell_columns + 1) / 2
    range_columns_start = -(cell_columns / 2)

    # interval of bin for the histogram of each cell (20° for each bin)
    number_of_orientations_per_180 = (180. * ABS_ANGLE) / number_of_orientations

    for i in range(number_of_orientations):
        # isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        c = c_0
        r = r_0
        r_i = 0
        c_i = 0

        while r < cc:
            c_i = 0
            c = c_0

            while c < cr:
                # get angle in the interval with the magnitude to indicate how we care about the information
                orientation_histogram[r_i, c_i, i] = \
                    cell_hog(magnitude, orientation,
                             orientation_start, orientation_end,
                             int(cell_columns), int(cell_rows), int(c), int(r),
                             int(size_columns), int(size_rows),
                             int(range_rows_start), int(range_rows_stop),
                             int(range_columns_start), int(range_columns_stop))
                c_i += 1
                c += cell_columns

            r_i += 1
            r += cell_rows


"""
Allows to get the feature vector of the Histogram Oriented Diagram of an image 
Source : https://github.com/scikit-image/scikit-image/blob/fe96435877f40581d678b10fde650c6e1899354a/skimage/feature/_hog.py
More readable version : https://github.com/trinhngocthuyen/teach-myself-ml/blob/master/funda_ml/HOG.ipynb
"""
def sobel_hog(image: np.ndarray,
              orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
              transform_sqrt=False):
    """
    Compute HOG features of an image. Return a row vector
    """

    # Equalisation of the image to reduce the influence of noise and illumination
    if transform_sqrt:
        image = np.sqrt(image)

    g_row, g_col = compute_gradient(image)
    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations),
                                     dtype=float)
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    # get the histogram for each bloc
    hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                   n_cells_col, n_cells_row,
                   orientations, orientation_histogram)

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, orientations)
    )

    # normalize the result of each block
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = hog_normalize_block(block)

    # return the feature vector
    return normalized_blocks.ravel()


"""
Computes the histogram of oriented gradient of an image with sobel or classic filter
Source : https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f#:%7E:text=Histogram%20of%20Oriented%20Gradients%2C%20also,the%20purpose%20of%20object%20detection
"""
def get_image_and_hist(file, path, im_width, im_height):
    image = cv2.imread(path + "/" + file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if DEBUG:
        plt.imshow(image)
        plt.show()

        height = image.shape[0]
        width = image.shape[1]

        print('Image Height       : ', height)
        print('Image Width        : ', width)

    image = cv2.resize(image, (im_width, im_height))

    if SOBEL_FILTER:
        # print("Sobel")
        fd = sobel_hog(image, orientations=9*ABS_ANGLE, pixels_per_cell=(CELL_SIZE, CELL_SIZE),
                       cells_per_block=(BLOCK_SIZE, BLOCK_SIZE), transform_sqrt=True)
    else:
        fd, hog_image = hog(image, orientations=9*ABS_ANGLE, pixels_per_cell=(CELL_SIZE, CELL_SIZE),
                            cells_per_block=(BLOCK_SIZE, BLOCK_SIZE), visualize=True,
                            transform_sqrt=True, block_norm='L2', feature_vector=True)

    if DEBUG:
        print(fd.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')

        plt.show()

    return image, fd


"""
Allows to get the feature vector of HOG for the reference images
"""
def create_reference_hist():
    files = os.listdir(BANQUE_IMAGES_PATH)
    feature_vectors = []

    for file in files:
        image, feature_vector = get_image_and_hist(file, BANQUE_IMAGES_PATH, WIDTH, HEIGHT)

        feature_vectors.append({
            "image": file,
            "feature_vector": feature_vector
        })

    return feature_vectors

"""
# Source : https://github.com/SamPlvs/Object-detection-via-HOG-SVM/blob/master/testing_HOG_SVM.py
def sliding_window(image, stepSize,
                   windowSize):  # image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0],
                   stepSize):  # this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
"""

"""
Allows for each request images to get the HOG feature vector of each color axis and to compare it 
to feature vectors that have been found for reference images thanks to euclidian distance. 
It will do the ranking and save the result in a file in order to get the precision of our prediction
"""
def image_recognition(descriptors, time_create_descritor_ref):
    files = os.listdir(REQUEST_IMAGES_PATH)
    test_pass = 0

    for file in files:
        start = time.time()

        # EXPLORATION
        # image_resize = cv2.imread(REQUEST_IMAGES_PATH + "/" + file)
        # image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
        # image_resize = cv2.resize(image_resize, (200, 300))

        image, feature_vector = get_image_and_hist(file, REQUEST_IMAGES_PATH, WIDTH, HEIGHT)
        distances = {}
        # For each reference we compute the euclidian distance
        for reference in descriptors:
            feature_vector_ref = reference['feature_vector']

            assert len(feature_vector_ref) == len(feature_vector)

            # Euclidian distance
            min_dist = np.linalg.norm(feature_vector - feature_vector_ref)

            """
            # EXPLORATION
            # Source : https://github.com/SamPlvs/Object-detection-via-HOG-SVM/blob/master/testing_HOG_SVM.py
            (winW, winH) = (64, 128)
            for resized in pyramid_gaussian(image_resize, downscale=1.5):  # loop over each layer of the image that you take!
                for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
                    if window.shape[0] != winH or \
                            window.shape[1] != winW:  # ensure the sliding window has met the minimum size requirement
                        continue

                    fd, hog_image = hog(window, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), visualize=True,
                                        transform_sqrt=True, block_norm='L2', feature_vector=True)

                    dist = np.linalg.norm(fd - feature_vector_ref)

                    if dist < min_dist:
                        min_dist = dist
            """

            distances[reference['image']] = min_dist

        # We order the result to have the best first
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))

        if DEBUG:
            print(distances)

        top = {}
        # We do the selection of the best predictions
        for i in range(0, TOP_VOTING):
            file_ref = list(distances.keys())[i]

            # The prediction can be None
            if distances[file_ref] <= THRESHOLD_DIST:
                top[file_ref] = distances[file_ref]

        if DEBUG:
            print(top)

        elements = {}
        # Thanks to the best predictions we can predict the association between the image and an object. Specify by
        # reference images name
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
        print(file, ' ', test_ok)
        end = time.time()
        exec_time = end - start

        # We save the result in a csv file
        utils.save_result(file, time_create_descritor_ref, exec_time, distances, top, winner, test_ok,
                          RESULT_HOG_FILE_PATH)

        if test_ok:
            test_pass = test_pass + 1

    precision = test_pass / len(files)
    print(precision)
    f = open(RESULT_HOG_FILE_PATH, "a")
    f.write("\nPrécision de l'algorithme ;" + str(precision) + "\n")
