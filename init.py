

import argparse
import sys
sys.path.append('cocoapi/PythonAPI/')
import socket
import ssl
hostname = socket.gethostname()
if 'shannon' in hostname or 'markov' in hostname:
    # Need this when downloading on university cluster...
    ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('punkt')

from pathconf import PathConfig
from embed import generate_glove_vectors
from vocabulary import build_vocab, save_vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create resized dataset and vocabulary.')
    parser.add_argument('--vocab', nargs='?', type=bool,
                        default=False, help='Build vocabulary.')
    parser.add_argument('--vocab_threshold', nargs='?', type=int,
                        default=6, help='Vocabulary frequency threshold.')
    parser.add_argument('--glove', nargs='?', type=bool,
                        default=False, help='Generate glove vectors.')
    args = parser.parse_args()

    if args.vocab:
        print('Building vocabulary...')
        vocab = build_vocab(args.vocab_threshold)
        save_vocab(vocab)
        print(f'Vocabulary saved to {PathConfig.vocab_file}.')

    if args.glove:
        # Don't forget to: "conda install -c anaconda bcolz"
        generate_glove_vectors()
