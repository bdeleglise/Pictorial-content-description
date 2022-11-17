import numpy as np

import color_hist
import time

from constant import RESULT_COLOR_HIST_FILE_PATH


def run():
    print("Processing ...")
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
    print("Ended")






if __name__ == '__main__':
    run()


