import glob
import cv2

from settings import load_settings
from video_processor import VideoProcessor


"""
Load a sequence of images and save the predictions
"""

settings = load_settings()
settings['n_frames'] = 1  # override setting because the images are not sequential

processor = VideoProcessor()

for file in glob.glob("./test_images/*.jpg"):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_img = processor.process_frame(image)

    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    filename = file.split('/')[-1]
    cv2.imwrite("./output_images/" + filename, out_img)
