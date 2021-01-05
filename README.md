# Image Captioning

## Setup

### NLTK Data
If on university cluster, run:

```mkdir nltk_data && python -m nltk.downloader punkt -d nltk_data```

Then, put 

```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

in every executable file.

### ResNet101 Model
If on university cluster, run

```wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth```

### Initialize

Run ```python main.py``` with necessarry arguments.
