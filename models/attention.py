import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms
from pytorch_pretrained_bert import BertTokenizer, BertModel # pip install pytorch-pretrained-bert

from vocabulary import END_TOKEN, PAD_TOKEN, START_TOKEN, Vocabulary
from embed import load_glove_vectors
from dataset import COCODataset
from models.encoder import EncoderAttention
from metric import AccumulatingMetric
from train_utils import clip_gradient
from checkpoint import save_checkpoint, load_checkpoint, unpack_checkpoint
from metric import get_eval_score

class SoftAttention(nn.Module):
    """Attention network."""

    def __init__(self, encoder_dim=2048, decoder_dim=512, attention_dim=512):
        """Initialize attention network.

        Args:
            encoder_dim (int): Feature size of encoded images.
            decoder_dim (int): Size of the decoder's RNN.
            attention_dim (int): Size of the attention network.
        """

        super(SoftAttention, self).__init__()

        # Linear layer to transform encoded image.
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        # Linear layer to transform decoder's output.
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        # Linear layer to calculate values to be softmax-ed.
        self.full_att = nn.Linear(attention_dim, 1)
        # ReLU layer before computing full attention.
        self.relu = nn.ReLU()
        # Softmax layer to calculate weights.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """Forward propagation.

        Args:
            encoder_out (torch.Tensor): Encoded images. A tensor of dimension (batch_size, num_pixels, encoder_dim).
            decoder_hidden (torch.Tensor): Previous decoder output. A tensor of dimension (batch_size, decoder_dim).

        Returns:
            Attention-weighted encoding, attention weights.
        """

        att_enc = self.enc_att(encoder_out)
        att_dec = self.dec_att(decoder_hidden)
        att = self.full_att(
            self.relu(att_enc + att_dec.unsqueeze(1))).squeeze(2)
        attention_weights = self.softmax(att)
        att_weighted_enc = (
            encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)
        return att_weighted_enc, attention_weights


class AttentionDecoderParams:
    attention_dim = 512
    decoder_dim = 512
    embed_size = 512  # Use 300 if glove and 768 if BERT.
    dropout = 0.5
    use_bert = False
    vocab = None  # Must override.

class AttentionDecoder(nn.Module):

    def __init__(self, device, params):
        """Initialize decoder.

        Args:
            params (AttentionDecoderParams): Parameters for decoder.
        """

        super(AttentionDecoder, self).__init__()

        assert isinstance(params, AttentionDecoderParams)
        assert isinstance(params.vocab, Vocabulary)

        self.device = device

        self.encoder_dim = 2048  # Set in stone.
        self.attention_dim = params.attention_dim
        self.embed_size = params.embed_size
        self.decoder_dim = params.decoder_dim
        self.vocab = params.vocab
        self.vocab_size = len(self.vocab)
        self.dropout = params.dropout

        self.use_bert = params.use_bert
        if self.use_bert:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            self.bert_model.eval()

        # Soft attention
        self.attention = SoftAttention(
            self.encoder_dim, self.decoder_dim, self.attention_dim)

        # Decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            self.embed_size + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # Initialize layers with uniform distribution for easier convergence.
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        # Default behaviour is to fine-tune embeddings. If using pre-trained embeddings
        # you might not want fine-tuning.
        self.fine_tune_embeddings(on=True)

    def load_pretrained_embeddins(self, embeddings):
        """Loads embedding layer with pre-trained embeddings.

        Args:
            embeddings: Pre-trained embeddings.
        """

        # Embeddings could the the glove_vectors for example
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

    def init_hidden_state(self, encoder_out):
        """Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Args:
            encoder_out (torch.Tensor): Encoded images. A tensor of dimension (batch_size, num_pixels, encoder_dim).

        Returns:
            Hidden state, cell state.
        """

        mean_enc_out = encoder_out.mean(dim=1)
        h = self.h_lin(mean_enc_out)
        c = self.c_lin(mean_enc_out)
        return h, c

    def _create_bert_embeddings(self, encoded_captions):
        embeddings = []
        for encoded_caption in encoded_captions:
            caption = ' '.join([self.vocab.i2w[token.item()] for token in encoded_caption])
            caption = u'[CLS] '+ caption
            
            tokenized_caption = self.bert_tokenizer.tokenize(caption)                
            indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_caption)
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

            with torch.no_grad():
                encoded_layers, _ = self.bert_model(tokens_tensor)

            bert_embedding = encoded_layers[11].squeeze(0)
            
            split_caption = caption.split()
            tokens_embedding = []
            j = 0

            for full_token in split_caption:
                curr_token = ''
                x = 0
                for i, _ in enumerate(tokenized_caption[1:]): # disregard CLS
                    token = tokenized_caption[i+j]
                    piece_embedding = bert_embedding[i+j]
                    
                    # full token
                    if token == full_token and curr_token == '' :
                        tokens_embedding.append(piece_embedding)
                        j += 1
                        break
                    else: # partial token
                        x += 1
                        
                        if curr_token == '':
                            tokens_embedding.append(piece_embedding)
                            curr_token += token.replace('#', '')
                        else:
                            tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                            curr_token += token.replace('#', '')
                            
                            if curr_token == full_token: # end of partial
                                j += x
                                break                            

            caption_embedding = torch.stack(tokens_embedding)
            embeddings.append(caption_embedding)
  
        embeddings = torch.stack(embeddings)
        return embeddings


    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """Forward propagation.

        Args:
            encoder_out (torch.Tensor): Encoded images. A tensor of dimension (batch_size, 14, 14, encoder_dim).
            encoded_captions (torch.Tensor): Encoded captions. A tensor of dimension (batch_size, max_caption_length).
            caption_lengths (list): Caption lengths (including vocabulary.START_TOKEN and vocabulary.END_TOKEN). 
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We won't decode at vocabulary.END_TOKEN position since we have finished
        # generating as soon as we generate vocabulary.END_TOKEN. So, decoding lengths
        # are caption_lengths - 1.
        decode_lengths = [caption_length -
                          1 for caption_length in caption_lengths]

        # Get max decode lengths.
        max_decode_lengths = max(decode_lengths)

        if self.use_bert:
            # Use BERT embeddings.
            embeddings = self._create_bert_embeddings(encoded_captions)
        else:
            # Use regular embeddings.
            embeddings = self.embedding(encoded_captions)

        # Initialize hidden and cell states of LSTM.
        h, c = self.init_hidden_state(encoder_out)

        # Initialize predictions tensor.
        predictions = torch.zeros(
            batch_size, max_decode_lengths, vocab_size).to(self.device)
            
        # Initialize weights tensor (softmax output in attention)
        attention_weights = torch.zeros(
            batch_size, max_decode_lengths, num_pixels).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            t_encoder_out = encoder_out[:batch_size_t]
            t_decoder_hidden = h[:batch_size_t]
            t_decoder_cell = c[:batch_size_t]

            attention_weighted_encoding, attention_weight = self.attention(
                t_encoder_out, t_decoder_hidden)

            gate = self.sigmoid(self.f_beta(t_decoder_hidden))
            attention_weighted_encoding = gate * attention_weighted_encoding

            batch_embeds = embeddings[:batch_size_t, t, :]
            cat_val = torch.cat(
                [batch_embeds.double(), attention_weighted_encoding.double()], dim=1)

            h, c = self.decode_step(
                cat_val.float(), (t_decoder_hidden.float(), t_decoder_cell.float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            attention_weights[:batch_size_t, t, :] = attention_weight

        # predictionss, sorted captions, decoding lengths, attention wieghts
        return predictions, encoded_captions, decode_lengths, attention_weights


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

    # Collate function for dataloader.
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(data):
        imgs, captions = zip(*data)

        imgs = torch.stack(imgs, dim=0)
        captions = pad_sequence(
            captions, batch_first=True, padding_value=pad_idx)
        caption_lengths = [len(caption) for caption in captions]

        return imgs, captions, caption_lengths

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
        encoder = EncoderAttention()

        # Encoder optimizer; None if not fine-tuning encoder.
        encoder_optimizer = torch.optim.Adam(
            params=filter(lambda param: param.requires_grad,
                          encoder.parameters()),
            lr=args.encoder_lr) if args.fine_tune_encoder else None

        # Decoder.
        decoder_params = AttentionDecoderParams()
        decoder_params.attention_dim = args.attention_dim
        decoder_params.decoder_dim = args.decoder_dim
        decoder_params.embed_size = args.embed_size
        decoder_params.dropout = args.decoder_dropout
        decoder_params.vocab = dataset.vocab
        decoder_params.use_bert = args.use_bert
        decoder = AttentionDecoder(device, decoder_params)
        if args.use_glove:
            glove = load_glove_vectors()
            decoder.load_pretrained_embeddins(glove)
        decoder.fine_tune_embeddings(args.fine_tune_embedding)

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
    criterion = nn.CrossEntropyLoss().to(device)

    decoder.train()
    encoder.train()

    num_batches = len(train_loader)
    epoch_losses = [] if not 'epoch_losses' in metrics else metrics['epoch_losses']
    for epoch in range(start_epoch, args.epochs):
        batch_losses = []

        accum_loss = AccumulatingMetric()
        accum_time = AccumulatingMetric()

        start = time.time()

        for batch_idx, (imgs, captions, caption_lengths) in enumerate(train_loader):

            # Move to GPU if available.
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Encode.
            img_features = encoder(imgs)

            # Decode.
            scores, captions_sorted, decode_lengths, attention_weights = decoder(
                img_features, captions, caption_lengths)

            # Since we decoded starting with a START_TOKEN, the targets are all words after START_TOKEN, up to END_TOKEN
            # (i.e. we remove START_TOKEN and include END_TOKEN).
            targets = captions_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads.
            # pack_padded_sequence is an easy trick to do this.
            scores = pack_padded_sequence(
                scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True).data

            # Compute loss.
            loss = criterion(scores, targets).to(device)

            # Add doubly stochastic attention regularization.
            loss += ((args.alpha_c - attention_weights.sum(dim=1)) ** 2).mean()

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

    print(f'Model {args.model_name} finished training for {args.epochs} epochs.')

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

            scores, captions_sorted, decode_lengths, attention_weights = decoder(img_features, captions, caption_lengths)
            targets = captions_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads.
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss.
            loss = criterion(scores_packed, targets_packed)
            loss += ((1. - attention_weights.sum(dim=1)) ** 2).mean()
            accum_loss.update(loss.item(), sum(decode_lengths))
            losses.append(loss.item())

            # References
            for j in range(targets.shape[0]):
                img_captions = targets[j].tolist() # validation dataset only has 1 unique caption per img
                # Remove <start>, <end> and <pad> tokens.
                cleaned_image_captions = [w for w in img_captions if w not in [vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN)]]  # remove pad, start, and end
                img_captions = list(map(lambda c: cleaned_image_captions, img_captions))
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                pred = p[:decode_lengths[j]]
                # Remove <start>, <end> and <pad> tokens.
                cleaned_pred = [w for w in pred if w not in [vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN)]]
                temp_preds.append(cleaned_pred) 
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(hypotheses) == len(references)

            print(f'loss: {accum_loss.avg()}')
            if batch_idx % args.print_freq == 0:
                print(f'Batch {batch_idx+1}/{num_batches}, Loss {accum_loss.avg():.4f}')

    metrics = get_eval_score(references, hypotheses)
    metrics['losses'] = losses

    end_time = time.time() - start_time
    print(f'Checkpoint {args.checkpoint} finished evaluation in {end_time:.4f} seconds.')

    return metrics
