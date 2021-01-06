# Image Captioning

## Setup

### Download data

You need to download all necessary data and put the data into expected directories. First, do ```mkdir -p cocoapi/images```. Then download, the dataset by running the following commands:

* ```wget -b -P ./cocoapi/images http://images.cocodataset.org/zips/train2014.zip```
* ```wget -b -P ./cocoapi/images http://images.cocodataset.org/zips/val2014.zip```
* ```wget -b -P ./cocoapi http://images.cocodataset.org/annotations/annotations_trainval2014.zip```

Finally, unzip the files.

### NLTK Data

We use nltk's word tokenizer. If on university cluster, run:

```mkdir nltk_data && python -m nltk.downloader punkt -d nltk_data```

### ResNet101 Model

The encoder uses a pre-trained CNN. If on university cluster, run

```wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth```

to download it.

### Initialize

Run ```python init.py --vocab True```.

### Confirm

Run ```python dataset.py```
