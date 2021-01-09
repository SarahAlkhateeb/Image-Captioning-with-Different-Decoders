import os


class PathConfig:
    """Global paths."""

    pkldata_dir = 'pkldata'
    vocab_file = os.path.join(pkldata_dir, 'vocab.pkl')

    anno_dir = os.path.join('cocoapi', 'annotations')
    train_anno_file = os.path.join(anno_dir, 'captions_train2014.json')
    val_anno_file = os.path.join(anno_dir, 'captions_val2014.json')
    img_dir = os.path.join('cocoapi', 'images')
    train_img_dir = os.path.join(img_dir, 'train2014')
    val_img_dir = os.path.join(img_dir, 'val2014')

    glove = 'glove.6B'
    glove_vectors = os.path.join(glove, 'glove_vectors.pkl')
