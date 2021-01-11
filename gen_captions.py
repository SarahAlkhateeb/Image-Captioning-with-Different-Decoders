import sys
import os
sys.path.append('cocoapi/PythonAPI/')
import argparse
import torch
from checkpoint import load_checkpoint, unpack_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Generate caption')
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint of trained model.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chkpt = load_checkpoint(device, args)
    _, encoder, decoder, encoder_optimizer, decoder_optimizer, metrics = unpack_checkpoint(
        chkpt)

    print(metrics)

if __name__ == '__main__':
    main()