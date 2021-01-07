
# TODO(Lisa)

# Take inspo from https://github.com/johanronkko/DIT245-project-other/blob/main/models/decoder.py
import time
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms

from vocabulary import PAD_TOKEN
from dataset import COCODataset
from models.encoder import Encoder
from metric import AccumulatingMetric
from train_utils import clip_gradient
from checkpoint import save_checkpoint, load_checkpoint, unpack_checkpoint


class BaselineDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,encoder_dim=2048):
        """Initialize baseline model.
        

        Args:
            encoder_dim (int): Feature size of encoded images.
            embed_size (int): Size of the Embeddings.
            hidden_size (int): Size of the hidden layer.
            vocab_size(int): Size of the Vocabulary.
        """
        super().__init__()

        self.embed_size = embed_size

        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size

        # Embedding layer that turns words into a vector of a specified size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False) # unidirectional LSTM

        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)

        # linear layer to transform encoder output to embedding dimension
        self.img_features_transform = nn.Linear(encoder_dim, embed_size)

    def forward(self, img_features, captions):
        """ Define the feedforward behavior of the model """
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]
                
        # Create embedded word vectors for each word in the captions
        embeddings = self.embedding(captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)
    
        # Stack the features and captions
        embeddings = torch.cat((img_features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)

        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, _ = self.lstm(embeddings) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)

        return outputs


def train(device, args):
    """Trains attention model.
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

    # Collate function for datalaoader.
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(data):
        imgs, captions = zip(*data)

        imgs = torch.stack(imgs, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)

        return imgs, captions

    # Dataloader.
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn)

    if args.checkpoint is None:
        # Initialize encoder/decoder models and optimizers.
        # Encoder.
        encoder = Encoder(args.embed_dim)

        # Encoder optimizer; None if not fine-tuning encoder.
        encoder_optimizer = torch.optim.Adam(
            params=filter(lambda param: param.requires_grad, encoder.parameters()), 
            lr=args.encoder_lr) if args.fine_tune_encoder else None

        # Create decoder.
        decoder = BaselineDecoder(args.embed_dim, args.decoder_dim, len(dataset.vocab))

        # Decoder optimier.
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda param: param.requires_grad, decoder.parameters()), 
            lr=args.decoder_lr)

        start_epoch = 0
        metrics = {}
    
    else:
        # Load encoder/decoder models and optimizers from checkpoint.
        chkpt = load_checkpoint(device, args)
        start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, metrics = unpack_checkpoint(chkpt)
        start_epoch += 1


    # Move to GPU if available.
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Criterion.
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab(PAD_TOKEN)).to(device)

    decoder.train()
    encoder.train()

    num_batches = len(train_loader)
    epoch_losses = [] if not 'epoch_losses' in metrics else metrics['epoch_losses']
    for epoch in range(start_epoch, args.epochs):
        batch_losses = []

        accum_loss = AccumulatingMetric()
        accum_time = AccumulatingMetric()

        start = time.time()

        for batch_idx, (imgs, captions) in enumerate(train_loader):

            # Move to GPU if available.
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Encode.
            img_features = encoder(imgs)

            # Decode.
            scores = decoder(img_features, captions)

            # Compute loss.
            loss = criterion(scores.reshape(-1, scores.shape[2]), captions.reshape(-1)).to(device)

            # Back propagation.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients.
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)
            
            # Update weights.
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Statistics.
            batch_losses.append(loss.item())
            accum_loss.update(loss.item())
            accum_time.update(time.time() - start)
            if batch_idx % args.print_freq == 0:
                print(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{num_batches}, Loss {accum_loss.avg():.4f}, Time: {accum_time.val:.4f}')

            # Reset start time.
            start = time.time()

        epoch_losses.append(batch_losses)

        # Save checkpoint.
        metrics = {
            'epoch_losses': epoch_losses
        }
        save_checkpoint(args, epoch, encoder, decoder, 
            encoder_optimizer, decoder_optimizer, metrics)

    print(f'Model {args.model_name} finished training for {args.epochs} epochs.')

    








