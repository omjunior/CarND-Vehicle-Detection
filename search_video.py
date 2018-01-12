from moviepy.editor import VideoFileClip
from video_processor import VideoProcessor

from settings import load_settings

"""
Process a video file and save the results
"""

processor = VideoProcessor()

settings = load_settings()

clip1 = VideoFileClip("./project_video.mp4")
# clip1 = VideoFileClip("./project_video.mp4").subclip(t_start=6, t_end=10)
# clip1 = VideoFileClip("./project_video.mp4").subclip(t_start=24, t_end=28)
# clip1 = VideoFileClip("./test_video.mp4")

proc_clip = clip1.fl_image(processor.process_frame)

proc_clip.write_videofile("output_video/project_video" + ("_debug" if settings['DEBUG'] else "") + ".mp4", audio=False)
# proc_clip.write_videofile("output_video/test_video.mp4", audio=False)
