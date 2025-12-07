"""
@filename: video_editing.py
@author: Stmxlt
@time: 2025-04-07
"""


import os
import re
import tempfile
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from tqdm import tqdm


class VideoNewsEditor:
    def __init__(self, txt_path, video_path, output_path, icon_path="input_files/icon.png", 
                 font_path='simhei.ttf', font_size=40, title_font_size=64):
        """
        Initialize the VideoNewsEditor class for adding news text overlays to videos.
        
        Inputs:
            txt_path: Path to the text file containing news content (title + body).
            video_path: Path to the base video file (with background replaced).
            output_path: Path to save the final edited video.
            icon_path: Path to the icon image for the video overlay (default: "input_files/icon.png").
            font_path: Path to the font file for text rendering (default: 'simhei.ttf').
            font_size: Font size for body text (default: 40).
            title_font_size: Font size for news title (default: 64).
        """
        print("Initializing video editor...")

        self.txt_path = txt_path
        self.video_path = video_path
        self.output_path = output_path
        self.icon_path = icon_path
        self.font_path = font_path
        self.font_size = font_size
        self.title_font_size = title_font_size

        self.temp_dir = tempfile.TemporaryDirectory()
        self.news_items = []
        self.video_clip = None
        self.video_size = (0, 0)
        self.text_content = ""
        self.title = ""
        self.body_sentences = []

        self._parse_text_file()
        self._load_video()

    def _parse_text_file(self):
        """
        Parse the text file to extract news title (first line) and body content (remaining lines).
        Split body content into sentences for overlay.
        """
        print(f"Parsing text file: {self.txt_path}")
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"Text file not found: {self.txt_path}")

        with open(self.txt_path, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        if not all_lines:
            raise ValueError("Text file is empty")

        self.title = all_lines[0]

        if len(all_lines) < 2:
            print("Warning: Text file contains only title, no body content")
            self.body_sentences = []
            return

        body_content = ' '.join(all_lines[1:])
        split_sentences = self._split_sentences(body_content)
        self.body_sentences = self._split_long_sentences(split_sentences)
        self.news_items.append({'type': 'text', 'content': self.body_sentences})
        print(f"Processed {len(self.body_sentences)} sentences from all paragraphs")

    def _split_sentences(self, text):
        """
        Split continuous text into individual sentences using basic punctuation.
        Remove punctuation marks after splitting.
        """
        pattern = r'([。，！？…,.!?])'
        sentences = re.split(pattern, text)

        merged_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in ['。', '，', '！', '？', '…', '.', ',', '!', '?']:
                sentence_without_punct = sentences[i].strip()
                if sentence_without_punct:
                    merged_sentences.append(sentence_without_punct)
                i += 2
            else:
                if sentences[i].strip():
                    merged_sentences.append(sentences[i].strip())
                i += 1

        return [s for s in merged_sentences if s]

    def _split_long_sentences(self, sentences, max_length=25):
        """
        Split sentences longer than max_length into roughly equal parts.
        """
        processed = []
        for sentence in sentences:
            if len(sentence) <= max_length:
                processed.append(sentence)
            else:
                parts = (len(sentence) + max_length - 1) // max_length
                part_length = len(sentence) // parts

                for i in range(parts):
                    start = i * part_length
                    end = start + part_length if i < parts - 1 else len(sentence)
                    processed.append(sentence[start:end])
        
        return processed

    def _load_video(self):
        """Load the base video and extract its size and duration."""
        print("Loading video...")
        self.video_clip = VideoFileClip(self.video_path)
        self.video_size = self.video_clip.size
        self.total_duration = self.video_clip.duration

    def _create_icon_white_block(self):
        """
        Create a white block with a resized icon (for video overlay).
        
        Returns:
            ImageClip: Icon white block clip with full video duration, positioned at the bottom-left.
        """
        bar_height = self.font_size + 40
        
        icon_img = Image.open(self.icon_path).convert("RGBA")
        
        icon_width, icon_height = icon_img.size
        scale = bar_height / icon_height
        new_icon_width = int(icon_width * scale)
        new_icon_height = bar_height
        icon_resized = icon_img.resize((new_icon_width, new_icon_height), Image.LANCZOS)
        
        white_block_width = new_icon_width
        white_block_height = bar_height
        
        white_block = Image.new('RGBA', (white_block_width, white_block_height), (255, 255, 255, 255))
        
        icon_x = (white_block_width - new_icon_width) // 2
        icon_y = 0
        white_block.paste(icon_resized, (icon_x, icon_y), icon_resized)
        
        white_block_path = os.path.join(self.temp_dir.name, "white_block_with_icon.png")
        white_block.save(white_block_path)
        
        return ImageClip(white_block_path).with_duration(self.total_duration) \
            .with_position(('left', self.video_size[1] - white_block_height))

    def _create_blue_bar(self, white_block_width):
        """
        Create a blue bar with news title (for video overlay), positioned next to the icon block.
        
        Inputs:
            white_block_width: Width of the icon white block (to align the blue bar).
        
        Returns:
            ImageClip: Blue bar clip with full video duration, positioned at the bottom.
        """
        bar_height = self.font_size + 40
        bar_width = self.video_size[0] - white_block_width
        
        img = Image.new('RGBA', (bar_width, bar_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype(self.font_path, self.title_font_size)
        except IOError:
            title_font = ImageFont.load_default()

        title_text = self.title if self.title else ""
        title_width = draw.textlength(title_text, font=title_font) if title_text else 0
        title_padding = 20
        title_total_width = int(title_padding + title_width)
        title_bg_end = min(title_total_width + 10, bar_width)

        for x in range(title_bg_end):
            draw.line(
                [(x, 0), (x, bar_height - 1)],
                fill=(0, 51, 153, 255)
            )

        gradient_start = title_bg_end
        gradient_length = bar_width - gradient_start
        
        if gradient_length > 0:
            for x in range(gradient_start, bar_width):
                alpha = int(255 * (1 - (x - gradient_start) / gradient_length))
                draw.line(
                    [(x, 0), (x, bar_height - 1)],
                    fill=(0, 51, 153, alpha)
                )

        if title_text:
            text_y = int((bar_height - self.title_font_size) // 2)
            
            # Add black shadow
            for x_offset in [-1, 1]:
                for y_offset in [-1, 1]:
                    draw.text(
                        (title_padding + x_offset, text_y + y_offset),
                        title_text,
                        font=title_font,
                        fill=(0, 0, 0, 255)
                    )
            # Add white text
            draw.text(
                (title_padding, text_y),
                title_text,
                font=title_font,
                fill=(255, 255, 255, 255)
            )
        
        bar_path = os.path.join(self.temp_dir.name, "blue_bar.png")
        img.save(bar_path)
        
        return ImageClip(bar_path).with_duration(self.total_duration) \
            .with_position((white_block_width, self.video_size[1] - bar_height))

    def _create_text_clip(self, text, start_time, duration):
        """
        Create a text clip for a single sentence, with shadow effect.
        
        Inputs:
            text: Sentence text to display.
            start_time: Time (in seconds) when the text starts to appear.
            duration: Duration (in seconds) for which the text is displayed.
        
        Returns:
            ImageClip: Text clip with specified start time and duration, centered horizontally.
        """
        img = Image.new('RGBA', (self.video_size[0], self.font_size + 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            font = ImageFont.load_default()

        text_width = draw.textlength(text, font=font)
        text_x = (self.video_size[0] - text_width) // 2
        text_y = 10
        position = (text_x, text_y)

        # Add black shadow
        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                draw.text(
                    (position[0] + x_offset, position[1] + y_offset),
                    text,
                    font=font,
                    fill=(0, 0, 0, 255)
                )
        # Add white text
        draw.text(position, text, font=font, fill=(255, 255, 255, 255))

        text_path = os.path.join(self.temp_dir.name, f"text_{start_time}.png")
        img.save(text_path)

        bar_height = self.title_font_size + self.font_size
        return ImageClip(text_path).with_duration(duration).with_start(start_time) \
            .with_position(('center', self.video_size[1] - bar_height - self.font_size - 10))

    def _process_timeline(self):
        """
        Build the video timeline by combining the base video, icon block, blue bar, and text clips.
        
        Returns:
            CompositeVideoClip: Final video clip with all overlays.
        """
        print("Building timeline...")
        clips = [self.video_clip]
        
        white_block_clip = self._create_icon_white_block()
        clips.append(white_block_clip)
        white_block_width = white_block_clip.size[0]
        
        blue_bar = self._create_blue_bar(white_block_width)
        clips.append(blue_bar)
        
        current_time = 0
        all_sentences = self.body_sentences
        
        total_chars = sum(len(sentence) for sentence in all_sentences)
        total_sentences = len(all_sentences)
        
        if total_chars == 0 or total_sentences == 0:
            print("No body content to process")
            return CompositeVideoClip(clips)
        
        char_duration = self.total_duration / total_chars
        print(f"Total duration: {self.total_duration:.2f}s, Total chars: {total_chars}")
        print(f"Base duration per character: {char_duration:.2f}s")

        for sentence in tqdm(all_sentences, desc="Processing sentences", unit="sentence"):
            sentence_len = len(sentence)
            duration = sentence_len * char_duration
            duration = min(duration, self.total_duration - current_time)
            
            if duration <= 0:
                break
            
            text_clip = self._create_text_clip(sentence, current_time, duration)
            clips.append(text_clip)
            
            current_time += duration

        return CompositeVideoClip(clips)

    def render_video(self):
        """
        Render the final edited video with all overlays (icon, title bar, body text) and save it.
        Clean up temporary files after rendering.
        """
        try:
            print("Rendering final video...")
            final_clip = self._process_timeline()
            final_clip.write_videofile(
                self.output_path,
                fps=self.video_clip.fps,
                codec="libx264",
                audio_codec="aac"
            )
            print(f"Video saved to: {self.output_path}")
        except Exception as e:
            print(f"Error rendering video: {e}")
        finally:
            self.temp_dir.cleanup()
            os.remove(self.video_path)
            if self.video_clip:
                self.video_clip.close()
