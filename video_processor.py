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
        acc_heatmap = np.copy(heatmap)

        bboxes = find_bboxes_from_heatmap(apply_threshold(heatmap, self.settings['heat_threshold']))

        if self.settings['DEBUG']:
            single_heatmap = np.clip(add_heat(np.zeros_like(img[:, :, 0]).astype(np.float),
                                              self.prev_frames[len(self.prev_frames)-1]), 0, 255)
            single_heatmap = np.dstack((single_heatmap, single_heatmap, single_heatmap))
            acc_heatmap = np.clip(acc_heatmap, 0, 255)
            acc_heatmap = np.dstack((acc_heatmap, acc_heatmap, acc_heatmap))
            from scipy.ndimage.measurements import label
            labels = np.clip(label(heatmap)[0], 0, 1)*255
            labels = np.dstack((labels, labels, labels))
            final = draw_boxes(img, bboxes)
            return np.concatenate((np.concatenate((single_heatmap, acc_heatmap), axis=1),
                                   np.concatenate((labels, final), axis=1)), axis=0)
        else:
            return draw_boxes(img, bboxes)
