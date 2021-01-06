import os
import sys
sys.path.append('cocoapi/PythonAPI/')
import argparse
import torch

from pathconf import PathConfig
from attention import train as train_attention_model
from baseline import train as train_baseline_model

def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model', type=str, choices=['baseline', 'attention'], help='Model to train')
    parser.add_argument('--attention_dim', type=int, default=512, help='attention dimension.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='decoder dimension.')
    parser.add_argument('--decoder_dropout', type=float, default=0.5, help='decoder dropout probability')
    parser.add_argument('--embed_dim', type=int, default=512, help='embedding dimension. If using pre-trained glove vectors, use 300.')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--workers', type=int, default=1, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1., help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=False, help='whether fine-tune encoder or not')
    parser.add_argument('--fine_tune_embedding', type=bool, default=False, help='whether fine-tune word embeddings or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--embedding_path', default=None, help='path to pre-trained word Embedding.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(PathConfig.vocab_file):
        raise SystemError('Must run python init.py --vocab before training.')
         
    if args.model == 'baseline':
        print('Training baseline model...')
        train_baseline_model(device, args)
        return
    
    if args.model == 'attention':
        print('Training attention model...')
        train_attention_model(device, args)
        return

if __name__ == '__main__':
    main()
    
        

    
