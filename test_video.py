from moviepy.editor import VideoFileClip
from video_processor import VideoProcessor


processor = VideoProcessor()

clip1 = VideoFileClip("./test_video.mp4")
proc_clip = clip1.fl_image(processor.process_frame)
proc_clip.write_videofile("output_video/test_video.mp4", audio=False)
