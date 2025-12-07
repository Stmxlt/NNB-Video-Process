# NNB-Video-Processing-Function

NNB-Video-Processing integrates advanced background matting and news-style video editing functionalities, leveraging open-source libraries (such as OpenCV, PyTorch, and MoviePy) and the Robust Video Matting (RVM) model to enable seamless video customization. It simplifies the creation of professional news videos with customized backgrounds and text overlays.

#### Directory Structure

```plaintext
project_root/
├── input_files/               # Input resources
│   ├── input_video.mp4        # Source video (with foreground to preserve)
│   ├── background.png         # Target background image
│   ├── news.txt               # News content (title + body text)
│   └── icon.png               # Icon for news overlay (bottom-left)
├── output_files/              # Generated files
│   ├── final_output_video.mp4 # Video with replaced background (pre-editing)
│   └── edited_video.mp4       # Final news video with overlays
├── utils/                     # RVM functions resources
├── video_background_matting.py # Background replacement logic
├── video_editing.py           # News overlay logic
└── __init__.py                # Main execution script
```

#### Project Overview

This toolkit automates two key video processing tasks:
* **Background Replacement**: Uses the RVM model to separate foreground (e.g., a presenter) from the original video background and replace it with a target image.
* **Video Editing**: Adds news-style overlays (title bar, body text, and icons) to the background-replaced video, generating a polished news clip.

#### You Need to Upload:
* an original broadcast video with a green screen (input_video.mp4)
* a background image to be used as the new backdrop (background.png)
* an image-text news article that contains the captions to be added(image-text news.docx)


#### Features
* **AI-Powered Background Matting**: Precisely segments foreground using the RVM model, supporting GPU/CPU acceleration.
* **Automatic Text Segmentation**: Splits news content into readable sentences using punctuation (。, ！, ？, ., !, ?) and automatic splitting of long sentences (over 25 characters).
* **Customizable Overlays**: Supports custom fonts, icons, background images, and text sizes for news elements.

#### Installation

* Python 3.10.8
* FFmpeg (required for video/audio processing; install guide)
* Required Python packages:

```bash
pip install -r requirements.txt
```

* Additional dependencies:
    - RVM model checkpoint: Download from [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) and place in the project root (e.g., rvm_mobilenetv3.pth).

#### Usage

1. Configure Inputs
Modify the CONFIG dictionary in __init__.py to specify paths and parameters:
```python
CONFIG = {
    "input_path": "input_files/input_video.mp4",       # Path to source video
    "background_path": "input_files/background.png",   # Path to target background
    "output_path": "output_files/final_output_video.mp4", # Path for background-replaced video
    "rvm_variant": "mobilenetv3",                      # RVM model variant (e.g., "mobilenetv3")
    "rvm_checkpoint": "rvm_mobilenetv3.pth",           # Path to RVM checkpoint file
    "rvm_device": "cuda"                               # Device ("cuda" for GPU, "cpu" for CPU)
}
```

2. Prepare News Content
Create input_files/news.txt with:
* First line: News title (displayed in the blue title bar).
* Subsequent lines: News body content (split into timed text overlays).

3. Run the Toolkit
Execute the main script to run both background replacement and news editing:

```bash
python __init__.py
```

#### Core Components

1. Video Background Matting: 
Handles foreground segmentation and background replacement using the RVM model.

##### Workflow:

* **Extract Video Parameters**: Reads input video dimensions, FPS, and metadata.
* **Initialize RVM Model**: Loads the pre-trained RVM model for foreground segmentation.
* **Generate Green Screen Video**: Uses RVM to separate the foreground and save it with a green background (temporary file).
* **Replace Green Screen**: Detects green screen areas and replaces them with the target background image, preserving the foreground.
    - **Add Audio**: Merges the original audio from the input video into the processed video.
    - **Cleanup**: Removes temporary files (green screen video, audio clips) to save space.

2. Video Editing: 
Adds news-style overlays to the background-replaced video.

##### Workflow:

* Parse News Content: Extracts title (first line) and body text from news.txt. The body is split into sentences using:
    - Chinese punctuation (e.g., 。, ！).
    - Manual splitting for long sentences (over 25 characters).
* Create Overlays:
    - Icon Block: A white block with a resized icon (bottom-left corner).
    - Title Bar: A blue gradient bar (adjacent to the icon) displaying the news title with a shadow effect.
    - Body Text Clips: Timed text overlays (centered) for each sentence, with duration proportional to character count.
3. Composite Video: Merges the base video, icon block, title bar, and text clips into the final output.

#### Note

* Ensure all input paths are correct to avoid FileNotFoundError.
* For CPU processing, set rvm_device: "cpu" (slower than GPU).
* Large videos may require longer processing time; progress bars indicate status.
* Customize font_size, title_font_size, and icon_path in VideoNewsEditor initialization for different styles.

#### Acknowledgement

Part of the code in this project is developed based on the open-source project [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) The original project adheres to the GPL-3.0 license.
