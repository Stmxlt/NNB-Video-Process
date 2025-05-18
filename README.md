# NNB-Video-Process

A video processing function for the New News Broadcasting system

**You Need to Upload:**
* An original broadcasting video with green screen.(input_video.mp4)
* A background image you need to replace.(background.png)
* A image-text news to add captions for the video.(image-text news.docx)

**The Function Will Do:**
* Replacing the green part of the input video.
* Automatically parsing the input document, adding subtitles and displaying pictures.

If you need to run this function, you have to build enviornment follows:

```shell
pip install -r requirements.txt
```

Certainly, you need to install language models for caption generation, such as Chinese language model:
```bash
python -m spacy download zh_core_web_sm
```
