import numpy as np

import color_hist
import time
import sys
import hog
from constant import RESULT_COLOR_HIST_FILE_PATH, RESULT_HOG_FILE_PATH


"""
To get the pictural content of an image and associate it with a type present in banque d_images that best match :
python main.py ALGO
ALGO is a string that specify the algorithm to used 
if -BOB we use the bob's algo -> the color histogram algorithm
if -GRA11 we use our algo -> the hog algorithm
"""
def run():
    if len(sys.argv) != 2:
        print('Error args')
        exit(1)

    print("Processing ...")

    if sys.argv[1] == "-BOB":
        print("Color histogram Algorithm ...")
        print("Computes reference hists ...")
        start_create_hist_ref = time.time()
        color_hists_ref = color_hist.create_reference_hist()
        end_create_hist_ref = time.time()
        time_create_hist_ref = end_create_hist_ref - start_create_hist_ref

        f = open(RESULT_COLOR_HIST_FILE_PATH, "w")
        f.write("Bob Algo result \n\n")
        f.close()

        print("Images recognition ...")
        color_hist.image_recognition(color_hists_ref, time_create_hist_ref)
    elif sys.argv[1] == "-GRA11":
        print("HOG Algorithm ...")
        print("Computes reference hists ...")
        start_create_descriptor = time.time()
        descriptors = hog.create_reference_hist()
        end_ceate_descriptor = time.time()
        time_create_descritor_ref = end_ceate_descriptor - start_create_descriptor

        f = open(RESULT_HOG_FILE_PATH, "w")
        f.write("HOG Algo result \n\n")
        f.close()

        print("Images recognition ...")
        hog.image_recognition(descriptors, time_create_descritor_ref)
    else:
        print('Error args')
        exit(1)
    print("Ended")






if __name__ == '__main__':
    run()


