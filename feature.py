import numpy as np
import cv2
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Extracts HOG (Histogram of Oriented Gradients) features
    :param img: The input image
    :param orient: The number of orientations on the histogram
    :param pix_per_cell: 2-tuple specifying the size of each cell
    :param cell_per_block: 2-tuple defining the area (in cells) over which normalization is performed
    :param vis: boolean to compute an image visualization as well
    :param feature_vec: boolean to unroll all features
    :return: The feature vector (and an image if vis)
    """
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features


def convert_color(img, cspace):
    """
    Converts color space from RGB to cspace
    :param img: The original image
    :param cspace: The target color space
    :return: The converted image
    """
    if cspace == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif cspace == 'RGB' or cspace == 'Keep':
        feature_image = np.copy(img)
    else:
        raise ValueError("Unknown color_space", cspace)
    return feature_image


def bin_spatial(img, size=(32, 32)):
    """
    Computes binned color features
    :param img: The target color image (any color space)
    :param size: Tuple with the size of binning
    :return: The feature vector
    """
    features = cv2.resize(img, size).ravel()
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Computes color histogram features for each channel
    :param img: The target color image
    :param nbins: The number of bins for each channel histogram
    :param bins_range: The range of the pixel values
    :return: The feature vector for all channels concatenated
    """
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_features(img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extracts spatial, histogram and HOG features and combine them all in a feature vector
    :param img: The target image
    :param cspace: The color space to work the image
    :param spatial_size: For bin_spatial
    :param hist_bins: For color_hist
    :param orient: For get_hog_features
    :param pix_per_cell: For get_hog_features
    :param cell_per_block: For get_hog_features
    :param hog_channel: For get_hog_features
    :param spatial_feat: Bool for use spatial binning
    :param hist_feat: Bool for use color histogram binning
    :param hog_feat: Bool for use HOG features
    :return: A concatenated features vector
    """
    file_features = []
    feature_image = convert_color(img, cspace)

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell,
                                                     cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell,
                                            cell_per_block, vis=False, feature_vec=True)
        file_features.append(hog_features)
    return np.concatenate(file_features)
