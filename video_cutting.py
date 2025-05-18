"""
@filename: video_cutting.py
@author: Stmxlt
@time: 2025-04-07
"""

import os
import re
import tempfile
import random
import numpy as np
import spacy
from docx import Document
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from moviepy import *

class VideoNewsEditor:
    def __init__(self, docx_path, video_path, output_path,
                 font_path='simhei.ttf', image_duration=3, font_size=20):
        """Initialize the video editor"""
        print("Initializing video editor...")

        self.docx_path = docx_path
        self.video_path = video_path
        self.output_path = output_path
        self.font_path = font_path
        self.image_duration = image_duration
        self.font_size = font_size

        # Initialize temporary file system
        self.temp_dir = tempfile.TemporaryDirectory()
        self.news_items = []
        self.video_clip = None
        self.video_size = (0, 0)
        self.doc = None

        # Processing steps
        self._parse_document()
        self._load_video()

    def _parse_document(self):
        """Parse the Word document (maintain original paragraph order)"""
        print("Parsing Word document...")
        self.doc = Document(self.docx_path)

        for idx, para in enumerate(self.doc.paragraphs):
            # Process image paragraphs
            images = self._extract_images(para)
            if images:
                self.news_items.append({'type': 'image', 'content': images})
                continue

            # Process text paragraphs
            text = para.text.strip()
            if text:
                sentences = self._split_sentences(text)
                self.news_items.append({'type': 'text', 'content': sentences})

        print(f"Parsing complete, extracted {len(self.news_items)} news items")

    def _extract_images(self, paragraph):
        """Extract images from the paragraph"""
        images = []
        for run in paragraph.runs:
            for graphic in run.element.xpath('.//a:graphic'):
                blip = graphic.xpath('.//a:blip/@r:embed')
                if blip:
                    image_part = self.doc.part.related_parts[blip[0]]
                    img_path = os.path.join(self.temp_dir.name,
                                            f"img_{len(self.news_items)}.png")
                    with open(img_path, 'wb') as f:
                        f.write(image_part._blob)
                    images.append(img_path)
        if images:
            print(f"Image paths: {images}")
        return images

    def _split_sentences(self, text):
        """
        Split the input text into sentences and process each sentence to meet specific length requirements.

        Args:
            text (str): The input string to be processed.

        Returns:
            list: A list of processed sentences.
        """
        # Regular expression pattern to match Chinese punctuation marks
        pattern = r'([，、。！？…])'

        # Split the text into sentences based on punctuation marks
        sentences = re.split(pattern, text)

        # Merge sentences with their corresponding punctuation marks (since split separates punctuation into individual elements)
        merged_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in ['，', '、', '。', '！', '？', '…']:
                merged_sentences.append(sentences[i] + sentences[i + 1])
                i += 2
            else:
                merged_sentences.append(sentences[i])
                i += 1

        # Remove the trailing punctuation mark from each sentence
        cleaned_sentences = []
        for sentence in merged_sentences:
            # If the sentence ends with a punctuation mark, remove the last character
            if sentence and sentence[-1] in ['，', '、', '。', '！', '？', '…']:
                cleaned_sentences.append(sentence[:-1])
            else:
                cleaned_sentences.append(sentence)

        # Load the Chinese spaCy model
        try:
            nlp = spacy.load("zh_core_web_sm")
        except:
            print("Please install the Chinese spaCy model using: python -m spacy download zh_core_web_sm")
            nlp = None

        # Process sentences using spaCy for further splitting
        final_sentences = []
        for sentence in cleaned_sentences:
            if nlp is not None:
                # Use spaCy to split the sentence into clauses
                doc = nlp(sentence)
                clauses = [str(sent) for sent in doc.sents]
                if len(clauses) > 1:
                    # If spaCy split the sentence into multiple clauses, use them
                    final_sentences.extend(clauses)
                    continue
            final_sentences.append(sentence)

        # Further process sentences to ensure none exceed 25 characters
        processed_sentences = []
        for sentence in final_sentences:
            current_sentence = sentence
            while len(current_sentence) > 25:
                if nlp is not None:
                    # Try to split the sentence using spaCy again
                    doc = nlp(current_sentence)
                    clauses = [str(sent) for sent in doc.sents]
                    if len(clauses) > 1:
                        # If spaCy can split the sentence, use the split clauses
                        processed_sentences.extend(clauses)
                        current_sentence = None
                        break
                    else:
                        # If spaCy cannot split further, revert to the original method
                        split_point = 25
                        processed_sentences.append(current_sentence[:split_point])
                        current_sentence = current_sentence[split_point:]
                else:
                    # If spaCy is not available, revert to the original method
                    split_point = 25
                    processed_sentences.append(current_sentence[:split_point])
                    current_sentence = current_sentence[split_point:]
            if current_sentence is not None and current_sentence:
                processed_sentences.append(current_sentence)

        # Remove empty strings from the list
        processed_sentences = [sentence for sentence in processed_sentences if sentence]

        return processed_sentences

    def _load_video(self):
        """Load and preprocess the video"""
        print("Loading video...")
        self.video_clip = VideoFileClip(self.video_path)
        self.video_size = self.video_clip.size
        self.total_duration = self.video_clip.duration

    def _create_text_clip(self, text, start_time, duration):
        """Generate a text subtitle clip (rendered using PIL)"""
        print(f"Creating text subtitle: '{text}'")

        # Create text image
        img = Image.new('RGBA', (self.video_size[0], self.font_size + 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text position
        text_width = draw.textlength(text, font=font)
        position = ((self.video_size[0] - text_width) // 2, 10)

        # Draw black outline
        outline_color = (0, 0, 0, 255)  # Black
        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                draw.text((position[0] + x_offset, position[1] + y_offset), text, font=font, fill=outline_color)

        # Draw white text
        draw.text(position, text, font=font, fill=(255, 255, 255, 255))

        # Save temporary file
        text_path = os.path.join(self.temp_dir.name, f"text_{start_time}.png")
        img.save(text_path)
        print(f"Text subtitle saved to: {text_path}")

        return ImageClip(text_path).with_duration(duration).with_start(start_time) \
            .with_position(('center', self.video_size[1] - 150))

    def _process_timeline(self):
        """Build the timeline sequence"""
        print("Building timeline...")
        clips = []
        current_time = 0
        last_text_start = 0
        last_text_duration = 0
        image_clip_ranges = []  # Record the time ranges of image clips
        last_item_type = None  # Track the type of the last processed news item
        image_pool = []  # Collect all image paths from consecutive image news items

        # Calculate character density
        total_chars = sum(len(s) for item in self.news_items
                          if item['type'] == 'text' for s in item['content'])
        char_per_sec = total_chars / self.total_duration if total_chars else 0
        print(f"Character density: {char_per_sec} characters/second")

        # Process each news item
        for idx, item in enumerate(self.news_items):
            print(f"Processing news item {idx + 1}/{len(self.news_items)}")
            if item['type'] == 'text':
                # If there are collected images, process them before processing text
                if image_pool and last_item_type == 'image':
                    # Randomly select an image from the collected images
                    selected_img_path = random.choice(image_pool)
                    print(f"Randomly selected image: {selected_img_path}")

                    # Create image clip
                    img_clip = ImageClip(selected_img_path) \
                        .resized(width=self.video_size[0]) \
                        .with_duration(last_text_duration) \
                        .with_start(last_text_start) \
                        .with_position('center')

                    # Add fade-in and fade-out effects
                    fade_duration = last_text_duration * 0.15
                    img_clip = img_clip.with_effects([vfx.FadeIn(fade_duration), vfx.FadeOut(fade_duration)])
                    clips.append(img_clip)

                    # Record the time range of the image clip
                    image_clip_ranges.append((last_text_start, last_text_start + last_text_duration))

                    current_time = last_text_start + last_text_duration
                    image_pool = []  # Reset image pool after processing

                # Process text sentences
                for sentence in item['content']:
                    duration = max(len(sentence) / char_per_sec, 1) if char_per_sec else 0
                    print(f"Processing text sentence: '{sentence}', duration: {duration} seconds")
                    text_clip = self._create_text_clip(sentence, current_time, duration)
                    clips.append(text_clip)
                    last_text_start = current_time
                    last_text_duration = duration
                    current_time += duration
                last_item_type = 'text'  # Update the last news item type to text
            elif item['type'] == 'image':
                # Collect image paths
                if item['content']:
                    image_pool.extend(item['content'])
                last_item_type = 'image'  # Update the last news item type to image

        # Process any remaining images in the pool after the loop
        if image_pool and last_item_type == 'image':
            # Randomly select an image from the collected images
            selected_img_path = random.choice(image_pool)
            print(f"Randomly selected image: {selected_img_path}")

            # Create image clip
            img_clip = ImageClip(selected_img_path) \
                .resized(width=self.video_size[0]) \
                .with_duration(last_text_duration) \
                .with_start(last_text_start) \
                .with_position('center')

            # Add fade-in and fade-out effects
            fade_duration = last_text_duration * 0.15
            img_clip = img_clip.with_effects([vfx.FadeIn(fade_duration), vfx.FadeOut(fade_duration)])
            clips.append(img_clip)

            # Record the time range of the image clip
            image_clip_ranges.append((last_text_start, last_text_start + last_text_duration))

        print("Timeline building complete")
        return clips, image_clip_ranges  # Return the time ranges of image clips

    def render_video(self):
        """Render the final video"""
        print("Starting video rendering...")
        clips, image_clip_ranges = self._process_timeline()  # Get the time ranges of image clips
        print("Starting video composition...")

        # Create a dynamic background selection VideoClip
        def dynamic_bg(t):
            # Check if the current time is within the time range of an image clip
            for start, end in image_clip_ranges:
                if start <= t <= end:
                    # Apply Gaussian blur
                    frame = self.video_clip.get_frame(t)
                    img = Image.fromarray(frame)
                    img = img.filter(ImageFilter.GaussianBlur(radius=5))
                    return np.array(img)
            # Return the original video background if not in an image clip time range
            return self.video_clip.get_frame(t)

        # Create a dynamic background with the same size as the original video
        bg_frames = []
        for t in np.linspace(0, self.total_duration, int(self.total_duration * 24)):  # Assume 24fps
            bg_frames.append(dynamic_bg(t))

        # Combine the dynamic background frames into a new video clip
        bg_clip = ImageSequenceClip(bg_frames, fps=24)

        # Composite the video
        final = CompositeVideoClip([bg_clip] + clips, size=self.video_size)

        # Explicitly add audio
        final = final.with_audio(self.video_clip.audio)
        final.with_duration(self.total_duration).write_videofile(
            self.output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            logger=None  # Disable redundant logs
        )
        print(f"Video composition complete, saved to: {self.output_path}")
        # Delete the intermediate output files
        try:
            # Check if the file exist and delete
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
                print(f"Deleted file: {self.video_path}")
            else:
                print(f"File not found: {self.video_path}")
        except Exception as e:
            print(f"Error deleting files: {e}")

    def __del__(self):
        """Automatically clean up resources"""
        print("Cleaning up temporary resources...")
        self.temp_dir.cleanup()
        if self.video_clip:
            self.video_clip.close()
        print("Resource cleanup complete")