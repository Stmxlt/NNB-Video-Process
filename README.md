# NNB-Video-Processor

A video processing function for the New News Broadcasting system

**You Need to Upload:**
* an original broadcast video with a green screen (input_video.mp4)
* a background image to be used as the new backdrop (background.png)
* an image-text news article that contains the captions to be added(image-text news.docx)

**The Function Will:**
* replace the green area in the input video with the provided background
* automatically parse the input document, add subtitles, and display the accompanying images

If you need to run this function, you have to set up the environment as follows:

```shell
pip install -r requirements.txt
```

You also need to install language models for caption generation, such as Chinese language model:

```bash
python -m spacy download zh_core_web_sm
```
