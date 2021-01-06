import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision import transforms

from utils import clip_gradient
from dataset import COCODataset
from vocabulary import PAD_TOKEN
from encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    """Attention network."""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """Initialize attention network.

        Args:
            encoder_dim (int): Feature size of encoded images.
            decoder_dim (int): Size of the decoder's RNN.
            attention_dim (int): Size of the attention network.
        """

        super(Attention, self).__init__()

        self.enc_dim = encoder_dim
        self.dec_dim = decoder_dim
        self.att_dim = attention_dim

        # Linear layer to transform encoded image.
        self.enc_att = nn.Linear(self.enc_dim, self.att_dim)
        # Linear layer to transform decoder's output.
        self.dec_att = nn.Linear(self.dec_dim, self.att_dim)
        # Linear layer to calculate values to be softmax-ed.
        self.full_att = nn.Linear(self.att_dim, 1)
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
            Attention weighted encoding and weights (alpha).
        """
        
        att_enc = self.enc_att(encoder_out) # (batch_size_t, num_pizels, encoder_dim) -> (batch_size, num_pizels, attention_dim)
        att_dec = self.dec_att(decoder_hidden) # (batch_size_t, decoder_dim) -> (batch_size_t, attention_dim)
        att = self.full_att(self.relu(att_enc + att_dec.unsqueeze(1))).squeeze(2) # (batch_size_t, num_pixels, attention_dim) -> (batch_size_t, num_pixels)
        alpha = self.softmax(att) # (batch_size_t, num_pixels)
        att_weighted_enc = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch_size_1, encoder_dim)
        return att_weighted_enc, alpha

class Decoder(nn.Module):
    """RNN decoder with attention."""

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, pad_idx, encoder_dim=2048, dropout=0.5):
        """Initialize decoder.

        Args:
            attention_dim (int): Size of the attention network.
            embed_dim (int): Embedding size.
            decoder_dim (int): Size of the decoder's RNN.
            vocab_size (int): Size of vocabulary.
            pad_dix (int): Padding index used by vocabulary.
            encoder_dim (int): Feature size of encoded images.
            dropout: (float): Dropout probability of decoder's RNN.
        """

        super(Decoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout_prob = dropout

        # Attention network.
        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.att_dim)
        # Embedding layer.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Dropout layer.
        self.dropout = nn.Dropout(p=self.dropout_prob)
        # Decoding LSTMCell.
        self.lstm = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        # Linear layer to find initial hidden state of the LSTM cell.
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        # Linear layer to find initial cell state of LSTM cell.
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)
        # Linear layer to create a sigmoid-activated gate.
        self.f_beta = nn.Linear(self.encoder_dim, self.decoder_dim)
        # Sigmoid.
        self.sigmoid = nn.Sigmoid()
        # Linear layer to find scores over vocabulary.
        self.fc = nn.Linear(self.decoder_dim, vocab_size)

        # Initialize layers with the uniform distribution for easier convergence.
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

        self.fine_tune_embeddings()

    def fine_tune_embeddings(self, on=True):
        """Whether to allow fine-tuning of embedding layer. Only makes sense to not allow if using pre-trained embeddings. 
        Fine-tuning embeddings is the default behaviour of the decoder.

        Args:
            on (bool): Switch on or off.
        """

        for param in self.embedding.parameters():
            param.requires_grad = on

    def load_pretrained_embeddings(self, embeddings):
        """Loads embedding layer with pre-trained embeddings.

        Args:
            embeddings: Pre-trained embeddings.
        """

        self.embedding.weight = nn.Parameter(embeddings)

    def init_hidden_state(self, encoder_out):
        """Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Args:
            encoder_out (torch.Tensor): Encoded images. A tensor of dimension (batch_size, num_pixels, encoder_dim).

        Returns:
            Hidden state and cell state.
        """
        mean_enc_out = encoder_out.mean(dim=1) # (batch_size, num_pixels, encoder_dim) -> (batch_size, encoder_dim)
        h = self.init_h(mean_enc_out) # (batch_size, decoder_dim)
        c = self.init_c(mean_enc_out) # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """Forward propagation.

        Args:
            encoder_out (torch.Tensor): Encoded images. A tensor of dimension (batch_size, encoded_img_size, encoded_img_size, encoder_dim).
            encoded_captions (torch.Tensor): Encoded captions. A tensor of dimension (batch_size, max_caption_length).
            caption_lengths (torch.Tensor): Caption lengths. A tensor of dimension (batch_size, 1).

        Returns:
            Scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices.
        """

        # (batch_size, encoded_img_size, encoded_img_size, encoder_dim)/(batch_size, num_pixels, encoder_dim) -> (batch_size, num_pixels, encoder_dim)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image -> (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # For each data in the batch, when len(prediction) == len(caption_lengths), stop. 
        # Therefore, sort input data by decreasing lengths. 
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embeddings
        embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out) # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we have finished generating as soon as we generate <end>.
        # So, decoding lengths are atual lengths - 1.
        decode_lengths = (caption_lengths - 1).tolist()
        print(decode_lengths)
        print(encoded_captions)

        # Create tensors to hold word prediction scores and alphas.
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        print("hi")

        # At each time step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state output
        # and then generate a new word in the decoder with the previous word and the attention-weighted encoding.
        for t in range(max(decode_lengths)):
            print(t, max(decode_lengths))
            print("hi")


            batch_size_t = sum([l > t for l in decode_lengths])

            print(batch_size_t, num_pixels, encoder_dim)
            
            # alpha: (batch_size_t, num_pixels)
            # att_weighted_encoding: (batch_size_t, encoder_dim)
            att_weighted_enc, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            print(alpha.shape)
            print(att_weighted_enc.shape)
            print(h[:batch_size_t]) # [2, 512]

            # (encoder_dim, self.decoder_dim) == (2048, 512)
            print(self.f_beta(h[:batch_size_t]).shape)

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            att_weighted_enc = gate * att_weighted_enc
            h, c = self.lstm(torch.cat([embeddings[:batch_size_t, t, :], att_weighted_enc], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h)) # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alpha[:batch_size_t, t, :] = alpha
    
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class Params:
    # Must override
    vocab_size = "?" 
    pad_idx = "?"

    # Default
    attention_dim = 512
    embed_dim = 300
    decoder_dim = 512
    encoder_dim = 2048
    dropout = 0.5
    encoded_img_size=14
    attention_method='ByPixel'

def train(num_epochs, train_loader, model_params, fine_tune_encoder=False, fine_tune_embeddings=False, pretrained_embeddings=None):
    """Train attention model.
        
    Args:
        params (Params): Parameters to pass into encoder and decoder.
        fine_tune_encoder (bool): Whether to fine-tune encoder.
        fine_tune_embeddings (bool): Whether to fine-tune pre-trained embeddings.
        pretrained_embeddings: Pre-trained embeddings.
    """

    assert isinstance(model_params, Params)

    encoder = Encoder(model_params.encoded_img_size, model_params.attention_method)
    encoder.fine_tune(on=fine_tune_encoder)

    decoder = Decoder(model_params.attention_dim, model_params.embed_dim, model_params.decoder_dim, 
            model_params.vocab_size, model_params.pad_idx, model_params.encoder_dim, model_params.dropout)
    if pretrained_embeddings is not None:
        decoder.load_pretrained_embeddings(pretrained_embeddings)
    decoder.fine_tune_embeddings(on=fine_tune_embeddings)
    
    # Training parameters
    encoder_lr = 1e-4
    decoder_lr = 1e-4
    grad_clip = 5.
    alpha_c = 1.

    # Optimizers.
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):

            imgs, captions, caption_lengths = batch

            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            img_features = encoder(imgs)
            scores, captions_sorted, decode_lengths, alphas, sort_ind = decoder(img_features, captions, caption_lengths)
        
            # Since we decoded starting with a START_TOKEN, the targets are all words after START_TOKEN, up to END_TOKEN.
            targets = captions_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads.
            # pack_padded_sequence is an easy trick to do this.
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            scores = pack_padded_sequence(targets, decode_lengths, batch_first=True).data 

            # Compute loss.
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization.
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

            print(f'Loss: {loss.item()}')
            break

if __name__ == '__main__':
    num_epochs = 1
    batch_size = 2

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        normalize])

    dataset = COCODataset('train', img_transform, caption_max_len=25)
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab(PAD_TOKEN)

    def collate_fn(batch):
        batch = tuple(zip(*batch))
        imgs, captions, caption_lengths = batch[0][:], batch[1][:], batch[2][:]
        imgs = torch.stack(imgs)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        caption_lengths = torch.stack(caption_lengths)
        return imgs, captions, caption_lengths

    train_loader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)

    

    model_params = Params()
    model_params.vocab_size = vocab_size
    model_params.pad_idx = pad_idx

    train(num_epochs, train_loader, model_params)