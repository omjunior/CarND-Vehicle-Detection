from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
import pickle

from window import *


def proc_frame(img):
    windows = []
    for scale in scales:
        windows.extend(slide_window(img, x_start_stop=scale[1], y_start_stop=scale[2],
                                    xy_window=scale[0], xy_overlap=(0.75, 0.75)))

    hot_windows = search_windows(img, windows, clf, scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=4)
    return window_img


# TODO: Tweak these parameters and see how the results change.
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

scales = [[(80, 80), [None, None], [350, 550]],
          [(100, 100), [None, None], [350, 600]],
          [(120, 120), [None, None], [300, None]]]

clf = joblib.load("clf.p")
scaler = pickle.load(open("scaler.p", "rb"))

clip1 = VideoFileClip("./test_video.mp4")
proc_clip = clip1.fl_image(proc_frame)
proc_clip.write_videofile("output_video/test_video.mp4", audio=False)
