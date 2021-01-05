from pycocotools.coco import COCO
from collections import Counter
import nltk
import pickle

from pathconf import PathConfig

PAD_TOKEN = '<pad>' # Padding
START_TOKEN = '<start>' # Start of sentence
END_TOKEN = '<end>' # End of sentence
UNK_TOKEN = '<unk>' # Out of vocabulary (unknown)

class Vocabulary(object):
    """Represents vocabulary."""

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.w2i:
            self.w2i[word] = self.idx
            self.i2w[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.w2i:
            return self.w2i[UNK_TOKEN]
        return self.w2i[word]

    def __len__(self):
        return len(self.w2i)

def build_vocab(threshold=6):
    # Compute word frquencies from captions.
    coco = COCO(PathConfig.train_anno_file)
    counter = Counter()
    ids = coco.anns.keys()
    for id in ids:
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # Ommit non-frequent words determined by threshold.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create vocabulary.
    vocab = Vocabulary()
    vocab.add_word(PAD_TOKEN)
    for word in words:
        vocab.add_word(word)
    vocab.add_word(START_TOKEN)
    vocab.add_word(END_TOKEN)
    vocab.add_word(UNK_TOKEN)

    return vocab

def save_vocab(vocab):
    with open(PathConfig.vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab():
    with open(PathConfig.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
        return vocab