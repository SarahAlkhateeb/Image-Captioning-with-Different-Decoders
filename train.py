import sys
sys.path.append('cocoapi/PythonAPI/')
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms

from vocabulary import PAD_TOKEN
from dataset import COCODataset
from encoder import Encoder
from attention_decoder import AttentionDecoder, AttentionDecoderParams
from utils import clip_gradient

def train_baseline_model():
    pass

def train_attention_model(device, args):
    """Training attention model.

    Args:
        device: Device to run on.
        args: Parsed command-line arguments from argparse.
    """

    # Dataset.
    img_transform = transforms.Compose([
        transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    dataset = COCODataset(mode='train', img_transform=img_transform, caption_max_len=25)

    # Dataloader.
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(data):
        imgs, captions = zip(*data)

        imgs = torch.stack(imgs, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        caption_lengths = [len(caption) for caption in captions]

        return imgs, captions, caption_lengths

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn)

    # Create encoder.
    encoder = Encoder().to(device)

    # Encoder optimizer (None if not fine-tuning encoder).
    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda param: param.requires_grad, encoder.parameters()), 
        lr=args.encoder_lr) if args.fine_tune_encoder else None

    # Create decoder.
    decoder_params = AttentionDecoderParams()
    decoder_params.attention_dim = args.attention_dim
    decoder_params.decoder_dim = args.decoder_dim
    decoder_params.embed_dim = args.embed_dim
    decoder_params.dropout = args.decoder_dropout
    decoder_params.vocab_size = len(dataset.vocab)
    decoder = AttentionDecoder(device, decoder_params).to(device)
    decoder.fine_tune_embeddings(args.fine_tune_embedding)
    if args.embedding_path is not None:
        # TODO: handle pre-trained embeddings:
        pass

    # Decoder optimier.
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda param: param.requires_grad, decoder.parameters()), 
        lr=args.decoder_lr)

    # Criterion.
    criterion = nn.CrossEntropyLoss().to(device)

    decoder.train()
    encoder.train()

    for epoch in range(args.epochs):
        for batch_idx, (imgs, captions, caption_lengths) in enumerate(train_loader):

            # Move to GPU if available.
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Encode.
            img_features = encoder(imgs)

            # Decode.
            scores, captions_sorted, decode_lengths, attention_weights = decoder(img_features, captions, caption_lengths)

            # Since we decoded starting with a START_TOKEN, the targets are all words after START_TOKEN, up to END_TOKEN.
            targets = captions_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads.
            # pack_padded_sequence is an easy trick to do this.
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data 

            # Compute loss.
            loss = criterion(scores, targets).to(device)

            # Add doubly stochastic attention regularization.
            loss += ((args.alpha_c - attention_weights.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # Gradient clip decoder.
            clip_gradient(decoder_optimizer, args.grad_clip)
            
            decoder_optimizer.step()

            print(f'Loss: {loss.item()}')


def train_bert_model(device, args):
    pass

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

    if not args.model in ['baseline', 'attention']:
        print('Invalid model... exiting.')
        return
         
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
    
        

    
