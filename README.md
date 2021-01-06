# Image Captioning

## Setup

### Download data

* ```mkdir -p cocoapi/images```
* ```wget -b -P ./cocoapi/images http://images.cocodataset.org/zips/train2014.zip```
* ```wget -b -P ./cocoapi/images http://images.cocodataset.org/zips/val2014.zip```
* ```wget -b -P ./cocoapi http://images.cocodataset.org/annotations/annotations_trainval2014.zip```

Then, unzip the files.

### NLTK Data
If on university cluster, run:

```mkdir nltk_data && python -m nltk.downloader punkt -d nltk_data```

### ResNet101 Model
If on university cluster, run

```wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth```

### Initialize

Run ```python init.py --vocab True```.

### Confirm

Run ```python dataset.py```
