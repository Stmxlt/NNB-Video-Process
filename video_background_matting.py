"""
@filename: video_background_matting.py
@author: Stmxlt
@time: 2025-04-06
"""

import cv2
import numpy as np
import os
from moviepy import VideoFileClip

class VideoBackgroundMatting:
    def __init__(self, input_path, background_path, output_path="output_video.mp4", lower_green=None, upper_green=None):
        """
        Initialize the video background matting

        Args:
            input_path (str): Path to the input video file
            background_path (str): Path to the background image file
            output_path (str, optional): Path to the output video file. Defaults to "output_video.mp4".
            lower_green (numpy.ndarray, optional): Lower bound for green color in HSV. Defaults to None.
            upper_green (numpy.ndarray, optional): Upper bound for green color in HSV. Defaults to None.
        """
        self.input_path = input_path
        self.background_path = background_path
        self.output_path = output_path
        self.cap = None
        self.out = None
        self.background = None
        self.original_fps = None  # Store the original video frame rate
        self.video_width = None
        self.video_height = None
        # Default green range in HSV
        self.lower_green = np.array([50, 50, 50]) if lower_green is None else lower_green
        self.upper_green = np.array([77, 255, 255]) if upper_green is None else upper_green

    def load_video(self):
        """
        Load the video file and get the original frame rate
        """
        print("Loading video file...")
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video file: {self.input_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Debugging information
        print(f"Video loaded successfully:")

    def load_background(self):
        """
        Load the background image and crop it to match the video dimensions
        """
        self.background = cv2.imread(self.background_path)
        if self.background is None:
            raise Exception(f"Failed to load background image: {self.background_path}")

        # Get video dimensions
        video_width = self.video_width
        video_height = self.video_height

        # Get background image dimensions
        bg_height, bg_width = self.background.shape[:2]

        # Calculate video aspect ratio
        video_aspect_ratio = video_width / video_height

        # Calculate background image aspect ratio
        bg_aspect_ratio = bg_width / bg_height

        # Crop the background image based on aspect ratio
        if bg_aspect_ratio > video_aspect_ratio:
            # The background image has a larger aspect ratio than the video, crop the left and right sides
            new_width = int(bg_height * video_aspect_ratio)
            start_x = (bg_width - new_width) // 2
            self.background = self.background[:, start_x:start_x + new_width]
        else:
            # The background image has a smaller aspect ratio than the video, crop the top and bottom sides
            new_height = int(bg_width / video_aspect_ratio)
            start_y = (bg_height - new_height) // 2
            self.background = self.background[start_y:start_y + new_height, :]

        # Resize the background image to match the video dimensions
        self.background = cv2.resize(self.background, (video_width, video_height))

        # Check if the cropped background matches the video dimensions
        if self.background.shape[:2] != (video_height, video_width):
            raise Exception("The cropped background image dimensions do not match the video")

    def setup_output(self):
        """
        Set up the output video parameters using the original video frame rate
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.original_fps,
            (self.video_width, self.video_height)
        )

    def process_video(self):
        print("Processing video frames...")
        frame_count = 0
    
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
    
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}...")
    
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
            # Create mask
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
    
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
            # Smooth mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
            # Invert mask
            mask_inv = cv2.bitwise_not(mask)
    
            # Extract foreground and background
            foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)
            background = cv2.bitwise_and(self.background, self.background, mask=mask)
    
            # Combine foreground and background
            result = cv2.add(foreground, background)
    
            # Write result
            self.out.write(result)
    
        print(f"Processing complete. Total frames processed: {frame_count}")

    def release_resources(self):
        """
        Release resources
        """
        print("Releasing resources...")
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()

        # Debugging information
        print("Resources released successfully.")

    def add_audio(self):
        """
        Add audio to the output video and ensure the final video plays at the same rate as the original
        """
        try:
            print("Adding audio to the final video...")
            # Load the original video to extract audio
            original_video = VideoFileClip(self.input_path)
            audio = original_video.audio

            # Check if the original video has audio
            if audio is None:
                print("Original video does not contain audio. Skipping audio addition.")
                return

            # Load the output video (without audio)
            output_video = VideoFileClip(self.output_path)

            # Set the audio of the output video
            final_video = output_video.with_audio(audio)

            # Write the final video with audio to a new file
            final_output_path = "output_files/final_output_video.mp4"
            final_video.write_videofile(
                final_output_path,
                fps=self.original_fps,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True
            )

            print(f"Final video with audio saved to: {final_output_path}")

        except Exception as e:
            print(f"Error adding audio: {e}")

    def run(self):
        """
        Run the entire process
        """
        try:
            print("Starting background substitution process...")
            self.load_video()
            self.load_background()
            self.setup_output()
            self.process_video()
            self.release_resources()
            self.add_audio()
            print("Background substitution process completed successfully.")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.release_resources()
            # Delete the intermediate output files
            try:
                # Check if the file exist and delete
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                    print(f"Deleted file: {self.output_path}")
                else:
                    print(f"File not found: {self.output_path}")
            except Exception as e:
                print(f"Error deleting files: {e}")
