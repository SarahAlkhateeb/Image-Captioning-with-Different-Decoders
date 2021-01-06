import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms

from vocabulary import PAD_TOKEN
from dataset import COCODataset
from encoder import Encoder
from utils import clip_gradient

class SoftAttention(nn.Module):
    """Attention network."""

    def __init__(self, encoder_dim=248, decoder_dim=512, attention_dim=512):
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
        att = self.full_att(self.relu(att_enc + att_dec.unsqueeze(1))).squeeze(2)
        attention_weights = self.softmax(att)
        att_weighted_enc = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)
        return att_weighted_enc, attention_weights

class AttentionDecoderParams:
    attention_dim = 512
    decoder_dim = 512
    embed_dim = 512 # Use 300 if glove. 
    dropout = 0.5
    vocab_size = None # Must override.

class AttentionDecoder(nn.Module):

    def __init__(self, device, params):
        """Initialize decoder.

        Args:
            params (AttentionDecoderParams): Parameters for decoder.
        """

        super(AttentionDecoder, self).__init__()

        assert isinstance(params, AttentionDecoderParams)
        assert params.vocab_size is not None

        self.device = device

        self.encoder_dim = 2048 # Set in stone.
        self.attention_dim = params.attention_dim
        self.embed_dim = params.embed_dim
        self.decoder_dim = params.decoder_dim
        self.vocab_size = params.vocab_size
        self.dropout = params.dropout
        
        # soft attention
        self.attention = SoftAttention(self.encoder_dim, self.decoder_dim, self.attention_dim)

        # Decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # Embedding layer
        self.embedding = nn.Embedding(params.vocab_size, self.embed_dim)

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

        # embeddings could the the glove_vectors for example
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
        decode_lengths = [caption_length-1 for caption_length in caption_lengths]

        # Get max decode lengths.
        max_decode_lengths = max(decode_lengths)

        # Load embeddings.
        embeddings = self.embedding(encoded_captions)

        # Initialize hidden and cell states of LSTM.
        h, c = self.init_hidden_state(encoder_out)

        # Initialize predictions tensor.
        predictions = torch.zeros(batch_size, max_decode_lengths, vocab_size).to(self.device)
        # Initialize weights tensor (softmax output in attention)
        attention_weights = torch.zeros(batch_size, max_decode_lengths, num_pixels).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths ])
            
            t_encoder_out = encoder_out[:batch_size_t]
            t_decoder_hidden = h[:batch_size_t]
            t_decoder_cell = c[:batch_size_t]

            attention_weighted_encoding, attention_weight = self.attention(t_encoder_out, t_decoder_hidden)

            gate = self.sigmoid(self.f_beta(t_decoder_hidden))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            batch_embeds = embeddings[:batch_size_t, t, :]            
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1)
            
            h, c = self.decode_step(cat_val.float(),(t_decoder_hidden.float(), t_decoder_cell.float()))
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
