
import os
import json
import sys
sys.path.append('cocoapi/PythonAPI/')
import argparse
import torch
from checkpoint import load_checkpoint, unpack_checkpoint
import torch
import argparse
from models.attention import evaluate as evaulate_attention_model
from models.baseline import evaluate as evaulate_baseline_model
from pathconf import PathConfig

def save_eval_data(name, d):
    path = os.path.join(PathConfig.eval_data, f'{name}.json')
    with open(path, 'w+') as f:
        json.dump(d, f)


def main():
    parser = argparse.ArgumentParser(description='Generate caption')
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint of trained model.')
    parser.add_argument(
        '--model_type', type=str, choices=['baseline', 'attention'], help='Type of model to evaluate')
    parser.add_argument('--max_caption_length', type=int, default=-1,
                        help='only use captions with caption length <= 50 when training.')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size for beam search')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chkpt = load_checkpoint(device, args)
    _, encoder, decoder, _, _, _ = unpack_checkpoint(chkpt)

    if args.model_type == 'attention':
        metrics = evaulate_attention_model(device, args, encoder, decoder)
        print(metrics)
        save_eval_data(args.checkpoint.split('.')[0], metrics)
    else:
        metrics = evaulate_baseline_model(device, args, encoder, decoder)
        print(metrics)
        save_eval_data(args.checkpoint.split('.')[0], metrics)

if __name__ == '__main__':
    main()