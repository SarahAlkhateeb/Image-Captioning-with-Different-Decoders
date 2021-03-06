#this code was inpired by 
#https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program/blob/master/project_2_image_captioning_project/model.py
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from dataset import COCODataset
from models.encoder import Encoder
from metric import AccumulatingMetric
from train_utils import clip_gradient
from vocabulary import END_TOKEN, PAD_TOKEN, START_TOKEN, Vocabulary
from checkpoint import save_checkpoint, load_checkpoint, unpack_checkpoint
from embed import load_glove_vectors
from metric import get_eval_score


class BaselineDecoderParams:
    hidden_size = 512
    embed_size = 512 # Use 300 if glove.
    vocab_size = None # Must override.

class BaselineDecoder(nn.Module):
    def __init__(self, params):
        """Initialize baseline model.

        Args:
            params (BaselineDecoderParams): Parameters for decoder.
        """

        super().__init__()

        assert isinstance(params, BaselineDecoderParams)
        assert params.vocab_size is not None

        self.embed_size = params.embed_size

        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = params.hidden_size

        # Embedding layer that turns words into a vector of a specified size
        self.embedding = nn.Embedding(params.vocab_size, params.embed_size)

        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=params.embed_size,
                            hidden_size=params.hidden_size,  # LSTM hidden units
                            num_layers=1,  # number of LSTM layer
                            bias=True,  # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0,  # Not applying dropout
                            bidirectional=False)  # unidirectional LSTM

        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, the vocab_size.
        self.linear = nn.Linear(params.hidden_size, params.vocab_size)

    def load_pretrained_embeddins(self, embeddings):
        """Loads embedding layer with pre-trained embeddings.

        Args:
            embeddings: Pre-trained embeddings.
        """

        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, on=True):
        """Whether to allow fine-tuning of embedding layer.

        Only makes sense to not allow if using pre-trained embeddings. 
        Fine-tuning embeddings is the default behaviour of the decoder.

        Args:
            on (bool): Switch fine-tuning on or off.
        """

        for param in self.embedding.parameters():
            param.requires_grad = on

    def forward(self, img_features, captions):
        """Forward propagation.
        
        Args:
            img_features (torch.Tensor): Embedded image feature vectors of dimension (batch_size, embed_dim).
            captions (torch.Tensor): Encoded captions. A tensor of dimension (batch_size, max_caption_length).

        Returns:
            Caption scores of dimension (batch_size, caption length, vocab_size).
        """

        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]

        # Create embedded word vectors for each word in the captions
        # embeddings new shape : (batch_size, captions length - 1, embed_size)
        embeddings = self.embedding(captions)

        # Stack the features and captions
        # embeddings new shape : (batch_size, caption length, embed_size)
        embeddings = torch.cat((img_features.unsqueeze(1).float(), embeddings.float()), dim=1)

        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        # lstm_out shape : (batch_size, caption length, hidden_size)
        lstm_out, _ = self.lstm(embeddings)

        # Final output.
        outputs = self.linear(lstm_out)

        return outputs


def train(device, args):
    """Trains attention model.

    Args:
        device: Device to run on.
        args: Parsed command-line arguments from argparse.
    """

    # Dataset.
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    dataset = COCODataset(
        mode='train', img_transform=img_transform, caption_max_len=args.max_caption_length)

    # Collate function for datalaoader.
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(data):
        imgs, captions = zip(*data)

        imgs = torch.stack(imgs, dim=0)
        captions = pad_sequence(
            captions, batch_first=True, padding_value=pad_idx)

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
        encoder = Encoder(args.embed_size)

        # Encoder optimizer; None if not fine-tuning encoder.
        encoder_optimizer = torch.optim.Adam(
            params=filter(lambda param: param.requires_grad,
                          encoder.parameters()),
            lr=args.encoder_lr) if args.fine_tune_encoder else None

        # Create decoder.
        params = BaselineDecoderParams()
        params.embed_size = args.embed_size
        params.hidden_size = args.decoder_dim
        params.vocab_size = len(dataset.vocab)
        decoder = BaselineDecoder(params)
        if args.use_glove:
            glove = load_glove_vectors()
            decoder.load_pretrained_embeddins(glove)
        decoder.fine_tune_embeddings(on=args.fine_tune_embedding)

        # Decoder optimier.
        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda param: param.requires_grad,
                          decoder.parameters()),
            lr=args.decoder_lr)

        start_epoch = 0
        metrics = {}
    else:
        # Load encoder/decoder models and optimizers from checkpoint.
        chkpt = load_checkpoint(device, args)
        start_epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, metrics = unpack_checkpoint(
            chkpt)
        start_epoch += 1

    # Move to GPU if available.
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Criterion.
    criterion = nn.CrossEntropyLoss(
        ignore_index=dataset.vocab(PAD_TOKEN)).to(device)

    decoder.train()
    encoder.train()

    train_start = time.time()
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
            loss = criterion(
                scores.reshape(-1, scores.shape[2]), captions.reshape(-1)).to(device)

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
                print(
                    f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{num_batches}, Loss {accum_loss.avg():.4f}, Time: {accum_time.val:.4f}')

            # Reset start time.
            start = time.time()

        epoch_losses.append(batch_losses)

        # Save checkpoint.
        metrics = {
            'epoch_losses': epoch_losses
        }
        save_checkpoint(args, epoch, encoder, decoder,
                        encoder_optimizer, decoder_optimizer, metrics)

    train_time = time.time() - train_start
    print(f'Model {args.model_name} finished training for {args.epochs} epochs in {train_time:.4f} seconds.')


def evaluate(device, args, encoder, decoder):
    """Performs one epoch's evaluation.

    Args:
        device: Device to run on.
        args: Parsed command-line arguments from argparse.
        val_loader: DataLoader for validation data.
        encoder: Encoder model
        Decoder: Decoder model
    
    Returns:
        score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1., 'losses': []}
    """



    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    dataset = COCODataset(
        mode='val', img_transform=img_transform, caption_max_len=args.max_caption_length)

    # Collate function for dataloader.
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(data):
        imgs, captions, _, _ = zip(*data)

        imgs = torch.stack(imgs, dim=0)
        captions = pad_sequence(
            captions, batch_first=True, padding_value=pad_idx)
        caption_lengths = [len(caption) for caption in captions]

        return imgs, captions, caption_lengths

    # Dataloader.
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn)

    vocab = dataset.vocab

    criterion = nn.CrossEntropyLoss()

    references = [] 
    hypotheses = [] 
    
    decoder.eval()
    encoder.eval()

    accum_loss = AccumulatingMetric()
    losses = []

    # Batches
    num_batches = len(val_loader)
    start_time = time.time()
    print("Started validation...")
    with torch.no_grad():
        for batch_idx, (imgs, captions, caption_lengths) in enumerate(val_loader):
            
            # Forward prop.
            imgs = imgs.to(device)
            captions = captions.to(device)

            img_features = encoder(imgs)
            scores = decoder(img_features, captions)

            # Calculate loss.
            loss = criterion(
                scores.reshape(-1, scores.shape[2]), captions.reshape(-1))
            losses.append(loss.item())
            accum_loss.update(loss.item())

            # References
            for j in range(captions.shape[0]):
                img_captions = captions[j].tolist() # validation dataset only has 1 unique caption per img
                # Remove <start>, <end> and <pad> tokens.
                cleaned_image_captions = [w for w in img_captions if w not in [vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN)]]  # remove pad, start, and end
                img_captions = list(map(lambda c: cleaned_image_captions, img_captions))
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, pred in enumerate(preds):
                # Remove <start>, <end> and <pad> tokens.
                cleaned_pred = [w for w in pred if w not in [vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN)]]
                temp_preds.append(cleaned_pred) 
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(hypotheses) == len(references)

            if batch_idx % args.print_freq == 0:
                print(f'Batch {batch_idx+1}/{num_batches}, Loss {accum_loss.avg():.4f}')
    
    metrics = get_eval_score(references, hypotheses)
    metrics['losses'] = losses

    end_time = time.time() - start_time
    print(f'Checkpoint {args.checkpoint} finished evaluation in {end_time:.4f} seconds.')

    return metrics

