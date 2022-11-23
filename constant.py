BANQUE_IMAGES_PATH = "./banque d_images"
TOP_VOTING = 3
REQUEST_IMAGES_PATH = "./requete"

HIST_BIN = 16
HIST_RANGE = 256

# 2 for color_hists
# 20 for HOG
THRESHOLD_DIST = 20

DEBUG = False

RESULT_COLOR_HIST_FILE_PATH = "./result_color_hist.csv"
RESULT_HOG_FILE_PATH = "./result_hog.csv"
RESULT_CATEGORIES = {
    "requete_1.png": "pomme",
    "requete_2.png": "pomme",
    "requete_3.png": "tasse",
    "requete_4.png": "zebre",
    "requete_5.png": "banane",
    "requete_6.png": "planchesurf",
    "requete_7.png": "plancheneige",
    "requete_8.png": None,
}

# HOG
CELL_SIZE = 8  # Each cell is 8x8 pixels
BLOCK_SIZE = 2  # Each block is 2x2 cells
SOBEL_FILTER = True
ABS_ANGLE = 1
WIDTH = 64 #64
HEIGHT = 128 #128
