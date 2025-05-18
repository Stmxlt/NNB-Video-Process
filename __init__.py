"""
@filename: __init__.py
@author: Stmxlt
@time: 2025-04-09
"""

import sys
import os
import numpy as np

# Add the current directory to the Python path to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the two modules
from video_background_matting import VideoBackgroundMatting
from video_cutting import VideoNewsEditor


def main():
    """
    Main function to run both modules
    """
    # Run VideoBackgroundMatting
    substitutor = VideoBackgroundMatting(
        input_path="input_files/input_video.mp4",
        background_path="input_files/background.png",
        output_path="output_files/output_video.mp4",
        lower_green=np.array([50, 50, 50]),
        upper_green=np.array([140, 255, 255])
    )
    substitutor.run()

    # Run VideoNewsEditor
    editor = VideoNewsEditor(
        docx_path="input_files/image-text news.docx",
        video_path="output_files/final_output_video.mp4",
        output_path="output_files/edited_video.mp4",
        font_path="simhei.ttf",
        image_duration=3
    )
    editor.render_video()


if __name__ == "__main__":
    main()