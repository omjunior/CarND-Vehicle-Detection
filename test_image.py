import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
import glob
import cv2
import numpy as np

from settings import load_settings
from window import find_cars, draw_boxes


settings = load_settings()

clf = joblib.load("clf.p")
scaler = pickle.load(open("scaler.p", "rb"))

for file in glob.glob("./test_images/*.jpg"):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    draw_image = np.copy(image)

    found = []
    for scale in settings['scales']:
        found.extend(find_cars(image, scale[0], scale[1], scale[2], scale[3], scale[4], clf, scaler,
                               settings['color_space'], settings['orient'], settings['pix_per_cell'],
                               settings['cell_per_block'], settings['spatial_size'], settings['hist_bins']))

    window_img = draw_boxes(draw_image, found, color=(0, 0, 255), thick=4)

    plt.imshow(window_img)
    plt.show()
