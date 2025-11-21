"""
@filename: video_background_matting.py
@author: Stmxlt
@time: 2025-04-06
"""


import cv2
import numpy as np
import os
import torch
from typing import Tuple
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip
from utils.inference import convert_video
from utils.model import MattingNetwork
from utils.inference_utils import VideoReader


class VideoBackgroundMatting:
    def __init__(self, 
                 input_path: str, 
                 background_path: str, 
                 output_path: str = "output_files/final_output_video.mp4",
                 rvm_variant: str = "mobilenetv3",
                 rvm_checkpoint: str = "rvm_mobilenetv3.pth",
                 rvm_device: str = "cuda",
                 temp_green_screen_path: str = "output_files/temp_green_screen.mp4"):
        """
        Initialize the VideoBackgroundMatting class for video background replacement using RVM model.
        
        Inputs:
            input_path: Path to the input video file.
            background_path: Path to the target background image.
            output_path: Path to save the final output video (default: "output_files/final_output_video.mp4").
            rvm_variant: Variant of the RVM model (default: "mobilenetv3").
            rvm_checkpoint: Path to the RVM model checkpoint file (default: "rvm_mobilenetv3.pth").
            rvm_device: Device to run the RVM model (e.g., "cuda" or "cpu", default: "cuda").
            temp_green_screen_path: Path to save the temporary green screen video (default: "output_files/temp_green_screen.mp4").
        """
        self.input_path = input_path
        self.background_path = background_path
        self.output_path = output_path
        self.temp_green_screen_path = temp_green_screen_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(temp_green_screen_path), exist_ok=True)

        self.video_width, self.video_height, self.original_fps = self._get_video_params()

        self.background = self._load_and_adapt_background()

        self.rvm_model = self._init_rvm_model(rvm_variant, rvm_checkpoint, rvm_device)

    def _get_video_params(self) -> Tuple[int, int, float]:
        """
        Get basic parameters (width, height, FPS) of the input video.
        
        Returns:
            Tuple containing video width (int), height (int), and FPS (float).
        """
        try:
            video_reader = VideoReader(self.input_path)
            frame_height, frame_width, *_ = video_reader.video.frame_shape
            width = frame_width
            height = frame_height
            fps = video_reader.frame_rate
            del video_reader
            print(f"Input video parameters: {width}x{height} (width x height), FPS: {fps:.2f}")
            return width, height, fps
        except Exception as e:
            raise RuntimeError(f"Failed to get video parameters: {str(e)} (possible corrupted video file or wrong path)")

    def _load_and_adapt_background(self) -> np.ndarray:
        """
        Load the target background image and adapt it to match the input video's size.
        
        Returns:
            numpy.ndarray: Resized and formatted background image (BGR format).
        """
        background = cv2.imread(self.background_path)
        if background is None:
            raise FileNotFoundError(f"Target background image not found: {self.background_path}")
        
        background = cv2.resize(background, (self.video_width, self.video_height))
        
        if background.ndim == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        return background

    def _init_rvm_model(self, variant: str, checkpoint: str, device: str) -> torch.nn.Module:
        """
        Initialize the RVM (Robust Video Matting) model with the specified checkpoint.
        
        Inputs:
            variant: Model variant (e.g., "mobilenetv3").
            checkpoint: Path to the model checkpoint file.
            device: Device to load the model (e.g., "cuda" or "cpu").
        
        Returns:
            torch.nn.Module: Initialized RVM model in evaluation mode.
        """
        try:
            model = MattingNetwork(variant)
            checkpoint = torch.load(checkpoint, map_location=device)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model = model.eval().to(device)
            try:
                next(model.parameters())
            except StopIteration:
                raise RuntimeError(f"Model has no parameters after loading! Please check if the weight file: {checkpoint} is complete")
            
            print(f"RVM model initialized successfully! Device: {device}, Variant: {variant}")
            return model
        except FileNotFoundError:
            raise RuntimeError(f"RVM model weight file not found! Please confirm the path: {checkpoint} (official weights need to be downloaded in advance)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RVM model: {str(e)}")

    def _rvm_generate_green_screen(self):
        """
        Generate a green screen video (foreground with green background) using the RVM model.
        The result is saved to temp_green_screen_path.
        """
        print("Starting to generate green screen foreground video with RVM...")
        try:
            convert_video(
                model=self.rvm_model,
                input_source=self.input_path,
                output_type='video',
                output_composition=self.temp_green_screen_path,
                output_video_mbps=4,
                downsample_ratio=None,
                seq_chunk=12,
                progress=True
            )
            print(f"RVM green screen video generated: {self.temp_green_screen_path}")
        except Exception as e:
            raise RuntimeError(f"RVM failed to generate green screen video: {str(e)}")

    def _replace_green_screen_with_target(self) -> str:
        """
        Replace the green screen background in the temporary video with the target background image,
        while preserving the foreground (e.g., presenter).
        
        Returns:
            str: Path to the temporary video with replaced background (without audio).
        """
        print("Starting to replace green screen background with target image (preserving presenter)...")
        green_screen_cap = cv2.VideoCapture(self.temp_green_screen_path)
        if not green_screen_cap.isOpened():
            raise RuntimeError(f"Cannot read green screen video: {self.temp_green_screen_path}")
        
        ret, first_frame = green_screen_cap.read()
        if not ret:
            raise RuntimeError("Cannot read the first frame of green screen video, unable to locate green screen area")
        h, w = first_frame.shape[:2]
        green_roi = first_frame[h-50:h, w-50:w]
        avg_green_bgr = np.mean(green_roi, axis=(0, 1)).astype(np.uint8)
        print(f"Located green screen color (BGR): {avg_green_bgr} (only this color area will be replaced)")
        
        green_screen_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        final_no_audio_path = "output_files/temp_final_no_audio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        final_writer = cv2.VideoWriter(
            final_no_audio_path,
            fourcc,
            self.original_fps,
            (self.video_width, self.video_height)
        )
        if not final_writer.isOpened():
            raise RuntimeError(f"Cannot create video without audio: {final_no_audio_path}")
        
        h_offset = 5
        s_min = 60
        v_min = 60 

        avg_green_hsv = cv2.cvtColor(np.uint8([[avg_green_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        lower_green = np.array([max(0, avg_green_hsv[0] - h_offset), s_min, v_min])
        upper_green = np.array([min(179, avg_green_hsv[0] + h_offset), 255, 255])
        print(f"Green screen detection range: lower={lower_green}, upper={upper_green} (only green screen will be replaced)")
        
        kernel_close = np.ones((6, 6), np.uint8)
        kernel_blur = (5, 5)
        
        total_frames = int(green_screen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while green_screen_cap.isOpened():
                ret, frame = green_screen_cap.read()
                if not ret:
                    break
                
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                mask = cv2.GaussianBlur(mask, kernel_blur, 0)
                mask_norm = mask[:, :, np.newaxis] / 255.0
                
                foreground = frame * (1 - mask_norm)
                background = self.background * mask_norm
                final_frame = cv2.addWeighted(
                    foreground.astype(np.uint8), 1.0,
                    background.astype(np.uint8), 1.0,
                    gamma=0
                )
                
                final_writer.write(final_frame)
                pbar.update(1)
        
        green_screen_cap.release()
        final_writer.release()
        print(f"Background replacement completed! Video without audio: {final_no_audio_path} (presenter preserved)")
        return final_no_audio_path

    def _add_audio_to_final_video(self, no_audio_video_path: str):
        """
        Add audio from the original input video to the background-replaced video.
        
        Inputs:
            no_audio_video_path: Path to the video with replaced background (without audio).
        """
        print("Starting to add audio...")
        temp_audio_path = "output_files/temp_audio.m4a"
        try:
            original_audio = AudioFileClip(self.input_path)
            original_audio.write_audiofile(temp_audio_path, codec="aac")
            original_audio.close()
            
            no_audio_video = VideoFileClip(no_audio_video_path)
            temp_audio = AudioFileClip(temp_audio_path)
            
            final_video = no_audio_video.with_audio(temp_audio)
            
            final_video.write_videofile(
                self.output_path,
                fps=self.original_fps,
                codec="libx264",
                audio_codec="aac"
            )
            
            print(f"Audio added successfully! Final video: {self.output_path}")
        
        except Exception as e:
            error_msg = (
                f"Failed to add audio: {str(e)}\n"
                "2 Checks required: 1. Ensure ffmpeg is installed (required for MoviePy audio-video synthesis); "
                "2. output_files directory has write permission (right-click → Properties → Security → check write)"
            )
            raise RuntimeError(error_msg)
        
        finally:
            if 'no_audio_video' in locals():
                no_audio_video.close()
            if 'temp_audio' in locals():
                temp_audio.close()
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print(f"Temporary audio cleaned up: {temp_audio_path}")

    def _clean_temp_files(self):
        """Clean up temporary files generated during the process."""
        temp_files = [self.temp_green_screen_path, "output_files/temp_final_no_audio.mp4"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Temporary file cleaned up: {file}")

    def run(self):
        """
        Execute the entire background replacement workflow:
        1. Generate green screen video using RVM.
        2. Replace green screen with target background.
        3. Add original audio to the processed video.
        4. Clean up temporary files.
        """
        try:
            print("=" * 50)
            print("Starting RVM background replacement workflow...")
            self._rvm_generate_green_screen()
            no_audio_path = self._replace_green_screen_with_target()
            self._add_audio_to_final_video(no_audio_path)
            self._clean_temp_files()
            print("=" * 50)
            print(f"All processes completed! Final video saved to: {self.output_path}")
        except Exception as e:
            print(f"Workflow execution failed: {str(e)}")
            self._clean_temp_files()
            raise
