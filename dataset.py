from pathconf import PathConfig
from pycocotools.coco import COCO
from PIL import Image
from vocabulary import load_vocab, START_TOKEN, END_TOKEN
import torch.utils.data as data
import torch
import os
import nltk
import sys
sys.path.append('cocoapi/PythonAPI/')


class COCODataset(data.Dataset):
    def __init__(self, mode, img_transform=None, caption_max_len=50):
        assert mode in ['train', 'val']

        self.mode = mode
        self.img_transform = img_transform
        self.vocab = load_vocab()
        # Note, using a caption_max_len of 50 will half the training and validation sets.
        self.caption_max_len = caption_max_len
        self.anno_file = get_anno_file(mode)
        self.img_dir = get_img_dir(mode)
        self.coco = COCO(self.anno_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.caption_img_mappings = self._build_caption_img_mappings()

    def _build_caption_img_mappings(self):
        mappings = []
        for img_id in self.img_ids:
            anns = self._get_annotations(img_id)
            mapping = [{'caption': ann['caption'], 'img_id': img_id}
                       for ann in anns if len(ann['caption']) <= self.caption_max_len]
            mappings.extend(mapping)

        return mappings

    def _get_annotations(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        return anns

    def _numericalize_caption(self, caption):
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        result = []
        result.append(self.vocab(START_TOKEN))
        result.extend([self.vocab(token) for token in tokens])
        result.append(self.vocab(END_TOKEN))
        return torch.LongTensor(result)

    def _get_transformed_img(self, img_id):
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

    def __getitem__(self, idx):
        mapping = self.caption_img_mappings[idx]
        caption, img_id = mapping['caption'], mapping['img_id']

        img = self._get_transformed_img(img_id)
        caption = self._numericalize_caption(caption)

        # TODO: handle validation mode.

        return img, caption

    def __len__(self):
        # Number of captions in dataset. An image can have multiple alternative captions.
        return len(self.caption_img_mappings)


def get_anno_file(mode):
    if mode == 'train':
        return PathConfig.train_anno_file
    else:
        return PathConfig.val_anno_file


def get_img_dir(mode):
    if mode == 'train':
        return PathConfig.train_img_dir
    else:
        return PathConfig.val_img_dir


if __name__ == '__main__':
    # 11 captions with caption_max_len=25
    print(len(COCODataset('train', caption_max_len=25)))
    # 4 captions with caption_max_len=25
    print(len(COCODataset('val', caption_max_len=25)))
