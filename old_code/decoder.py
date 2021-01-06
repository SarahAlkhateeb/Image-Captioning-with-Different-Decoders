import torch
import torch.nn as nn

def make_embedding_layer(vocab_size, embed_size, use_glove):
    # TODO: extend to use glove embeddings if use_glove is True.
    return nn.Embedding(vocab_size, embed_size)

class DecoderLSTM(nn.Module):
    """DecoderLSTM

    Baseline decoder for our image captioning task.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, embed_dropout=0, use_glove=False):
        super(DecoderLSTM, self).__init__()
        self.embed = make_embedding_layer(vocab_size, embed_size, use_glove)
        self.dropout = nn.Dropout(embed_dropout)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, img_features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((img_features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

