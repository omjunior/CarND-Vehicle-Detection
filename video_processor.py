import pickle
import numpy as np
from sklearn.externals import joblib

from settings import load_settings
from window import find_cars, add_heat, apply_threshold, find_bboxes_from_heatmap, draw_boxes


class VideoProcessor:
    """
    A class to process frames of a video, taking into account previous frames
    """
    def __init__(self):
        """
        Constructor for VideoProcessor
        """
        self.prev_frames = []
        self.clf = joblib.load("clf.p")
        self.scaler = pickle.load(open("scaler.p", "rb"))
        self.settings = load_settings()
        return

    def process_frame(self, img):
        """
        Process the current frame
        :param img: Frame to be analysed
        :return: A list of bounding boxes containing cars
        """
        found = []
        for scale in self.settings['scales']:
            found.extend(find_cars(img, scale[0], scale[1], scale[2], scale[3], scale[4], self.clf, self.scaler,
                                   self.settings['color_space'], self.settings['orient'], self.settings['pix_per_cell'],
                                   self.settings['cell_per_block'], self.settings['spatial_size'],
                                   self.settings['hist_bins']))

        self.prev_frames.append(found)
        if len(self.prev_frames) > self.settings['n_frames']:
            self.prev_frames.pop(0)
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        for frame in self.prev_frames:
            heatmap = add_heat(heatmap, frame)

        bboxes = find_bboxes_from_heatmap(apply_threshold(heatmap, self.settings['heat_threshold']))

        return draw_boxes(img, bboxes)
