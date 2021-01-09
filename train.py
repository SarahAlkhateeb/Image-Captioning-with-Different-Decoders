import sys
sys.path.append('cocoapi/PythonAPI/')
import os
import argparse
import torch
from pathconf import PathConfig
from models.attention import train as train_attention_model
from models.baseline import train as train_baseline_model


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('model_name', type=str,
                        help='unique name of model setting; saved with this name in checkpoints folder.')
    parser.add_argument(
        '--model', type=str, choices=['baseline', 'attention'], help='Model to train')
    parser.add_argument('--attention_dim', type=int,
                        default=512, help='attention dimension.')
    parser.add_argument('--decoder_dim', type=int,
                        default=512, help='decoder dimension.')
    parser.add_argument('--decoder_dropout', type=float,
                        default=0.5, help='decoder dropout probability.')
    parser.add_argument('--embed_size', type=int, default=512,
                        help='embedding dimension. If using pre-trained glove vectors, use 300.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size.')
    parser.add_argument('--workers', type=int, default=1,
                        help='for data-loading.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                        help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float,
                        default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool,
                        default=False, help='whether to fine-tune encoder or not.')
    parser.add_argument('--fine_tune_embedding', type=bool, default=False,
                        help='whether to fine-tune word embeddings or not.')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='name of checkpoint in ./checkpoints folder; None if none.')
    parser.add_argument('--print_freq', type=int, default=1,
                        help='print training/validation stats every __ batches.')
    parser.add_argument('--use_glove', type=bool, default=False,
                        help='whether to use pre-trained glove embeddings.')
    parser.add_argument('--max_caption_length', type=int, default=50,
                        help='only use captions with caption length <= 50 when training.')
    parser.add_argument('--use_bert', type=bool,
                        default=False, help='whether to use BERT embeddigns for attention model.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(PathConfig.vocab_file):
        raise SystemError(
            'Must run "python init.py --vocab True" before training.')

    if args.use_glove:
        if not os.path.exists(PathConfig.glove_vectors):
            raise SystemError(
                'Must run "python init.py --glove True" when using glove vectors.')
        assert args.embed_size == 300, 'Expected embedding size of 300 for glove vectors.'

    if args.use_bert:
        assert args.model == 'attention', 'BERT is only used for attention model.'
        assert args.embed_size == 768, 'Expected embedding size of 768 for BERT.'

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
