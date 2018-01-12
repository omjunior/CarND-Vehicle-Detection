import cv2
import numpy as np
from scipy.ndimage.measurements import label

from features import color_hist, bin_spatial, convert_color, get_hog_features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draw bboxes over a copy of img
    :param img: The original image
    :param bboxes: The list of boxes to be drawn
    :param color: The color of the boxes
    :param thick: The thickness of the line
    :return: An image with the boxes drawn
    """
    draw_img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img


def find_cars(img, xstart, xstop, ystart, ystop, scale, clf, scaler, cspace, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, log=None, min_conf=0.4):
    """
    A function that find cars in a frame.
    It takes a region of interest of an image, converts it to a given color space, and runs a sliding window search
    through the image searching for matches by extracting a series of features and feeding them to the provided
    (trained) classifier, returning a list of found matches.
    The features computed are: HOG (using sub-sampling), spatial binning and color histogram.
    :param img: The input image
    :param xstart: Minimum X coordinate defining the region of interest
    :param xstop: Maximum X coordinate defining the region of interest
    :param ystart: Minimum Y coordinate defining the region of interest
    :param ystop: Maximum Y coordinate defining the region of interest
    :param scale: Defines the size of the window (scale * (64, 64))
    :param clf: The trained classifier
    :param scaler: A trained scaler used to normalize the features
    :param cspace: The target color space for the image before extracting features
    :param orient: The number of HOG orientations bins for the histogram
    :param pix_per_cell: 2-tuple specifying the size of each cell for extracting HOG
    :param cell_per_block:  2-tuple defining the area (in cells) over which normalization is performed during HOG
    :param spatial_size: 2-tuple defining the size for spatial binning
    :param hist_bins: The number of bins for each channel of color histogram
    :param log: A dictionary to log how many windows were searched and how many found cars
    :param min_conf: The minimum prediction confidence score to approve a prediction
    :return: A list of bounding boxes for every positive prediction from the classifier
    """
    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # With the definition below, an overlap of 75% is defined
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    window_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, spatial_size)
            hist_features = color_hist(subimg, hist_bins)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = clf.predict(test_features)
            conf = clf.decision_function(test_features)

            # Record prediction
            if log is not None:
                if scale not in log:
                    log[scale] = [0, 0]
                log[scale][1] += 1
                if test_prediction == 1 and conf > min_conf:
                    log[scale][0] += 1

            if test_prediction == 1 and conf > min_conf:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                window_list.append(((xbox_left + xstart, ytop_draw + ystart),
                                    (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))

    return window_list


def add_heat(heatmap, bbox_list):
    """
    For each box in bbox_list, increment belonging pixels
    :param heatmap:  The heatmap to accumulate (WILL BE CHANGED)
    :param bbox_list: A list of bounding boxes
    :return: The modified heatmap
    """
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Zero pixels that has been heated less than threshold
    :param heatmap: The heatmap to evaluate (WILL BE CHANGED)
    :param threshold: Threshold value to cool (inclusive)
    :return: The modified heatmap
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def find_bboxes_from_heatmap(heatmap):
    """
    Compute bounding boxes for each detection in a heatmap
    :param heatmap: The (thresholded) heatmap to search
    :return: A list of bounding boxes representing the location of each car
    """
    labels = label(heatmap)
    bboxes = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes
