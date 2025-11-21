"""
@filename: __init__.py
@author: Stmxlt
@time: 2025-04-09
"""


import sys
import os
from video_background_matting import VideoBackgroundMatting
from video_editing import VideoNewsEditor
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def main():
    """Main function to execute video background replacement and news video editing."""
    CONFIG = {
        "input_path": "input_files/input_video.mp4",
        "background_path": "input_files/background.png",
        "output_path": "output_files/final_output_video.mp4",
        "rvm_variant": "mobilenetv3",
        "rvm_checkpoint": "rvm_mobilenetv3.pth",
        "rvm_device": "cuda"
    }

    # Perform background matting
    background_matting = VideoBackgroundMatting(** CONFIG)
    background_matting.run()

    # Edit video with news content
    editor = VideoNewsEditor(
        txt_path="input_files/news.txt",
        video_path="output_files/final_output_video.mp4",
        output_path="output_files/edited_video.mp4",
        font_path="simhei.ttf",
        icon_path="input_files/icon.png",
        font_size=32,
        title_font_size=48
    )
    editor.render_video()


if __name__ == "__main__":
    main()
