def load_settings():
    """
    Provides a central place to define common settings
    :return: A dictionary with the settings
    """
    return {'color_space': 'YUV',  # All
            'orient':         11,  # HOG
            'pix_per_cell':   16,  # HOG
            'cell_per_block': 2,   # HOG
            'spatial_size':   (16, 16),  # Spatial binning
            'hist_bins':      32,  # Color histogram
            'scales':         [[200, 1080, 400, 500, 1],  # [ xmin, xmax, ymin, ymax, scale ]
                               [100, 1180, 400, 550, 1.5],
                               [0, 1280, 380, 600, 2],
                               [0, 1280, 380, 700, 3]],
            'n_frames':       5,  # Heat map
            'heat_threshold': 10  # Heat map
            }
