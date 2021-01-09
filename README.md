# Image Captioning

## Example run

From scratch:

```python train.py 'basic_att' --model 'attention' --batch_size 4 --epochs 3 --max_caption_length 25```

From checkpoint:

```python train.py 'basic_att' --model 'attention' --batch_size 4 --epochs 3 --checkpoint 'basic_att_1.pth.tar' --max_caption_length 25```

Using pre-trained glove embeddings:

```python train.py 'glove_baseline' --model 'baseline' --batch_size 4 --epochs 1 --use_glove True --fine_tune_embedding True --embed_size 300 --max_caption_length 25```

## Setup

## COCO API

Go to cocoapi/PythonAPI and run ```make```.

### Download data

You need to download all necessary data and put the data into expected directories. Download the dataset by running the following commands:

* ```wget -b -P ./cocoapi/images http://images.cocodataset.org/zips/train2014.zip```
* ```wget -b -P ./cocoapi/images http://images.cocodataset.org/zips/val2014.zip```
* ```wget -b -P ./cocoapi http://images.cocodataset.org/annotations/annotations_trainval2014.zip```

Finally, unzip the files (e.g. using ```unzip```).

Download glove embeddings by running:

```wget -P glove.6B http://nlp.stanford.edu/data/glove.6B.zip```

Then, unzip glove.6B.zip.

### NLTK Data

We use nltk's word tokenizer and store nltk data locally in this directory. Run:

```python -m nltk.downloader punkt -d nltk_data```

### ResNet101 Model

The encoder uses a pre-trained CNN. If on university cluster, stand in models dir and run 

```wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth```

to download it.

### Initialize

Run ```python init.py --vocab True```.

### Confirm

Run ```python dataset.py```


## References

* https://github.com/sankalp1999/Image_Captioning_With_Attention/blob/main/model.py
* https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/image_captioning/train.py
* https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program/blob/master/project_2_image_captioning_project/model.py
* https://github.com/ajamjoom/Image-Captions/blob/master/main.py
* https://github.com/RoyalSkye/Image-Caption/blob/master/train.py
