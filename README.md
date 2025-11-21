# NNB-Video-Processing-function

NNB-Video-Processing integrates advanced background matting and news-style video editing functionalities, leveraging open-source libraries (such as OpenCV, PyTorch, and MoviePy) and the Robust Video Matting (RVM) model to enable seamless video customization. It simplifies the creation of professional news videos with customized backgrounds and text overlays.

## Project Overview

This toolkit automates two key video processing tasks:
* **Background Replacement**: Uses the RVM model to separate foreground (e.g., a presenter) from the original video background and replace it with a target image.
* **Video Editing**: Adds news-style overlays (title bar, body text, and icons) to the background-replaced video, generating a polished news clip.

**You Need to Upload:**
* an original broadcast video with a green screen (input_video.mp4)
* a background image to be used as the new backdrop (background.png)
* an image-text news article that contains the captions to be added(image-text news.docx)


**Features**
* **AI-Powered Background Matting**: Precisely segments foreground using the RVM model, supporting GPU/CPU acceleration.
* **Automatic Text Segmentation**: Splits news content into readable sentences using Chinese punctuation and spaCy (if available).
* **Customizable Overlays**: Supports custom fonts, icons, background images, and text sizes for news elements.

If you need to run this function, you have to set up the environment as follows:

```shell
pip install -r requirements.txt
```

You also need to install language models for caption generation, such as Chinese language model:

```bash
python -m spacy download zh_core_web_sm
```
 Robust Video Matting (RVM). 

## Acknowledgement

Part of the code in this project is developed based on the open-source project [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) The original project adheres to the GPL-3.0 license, and this project is also open-sourced under the GPL-3.0 license.
