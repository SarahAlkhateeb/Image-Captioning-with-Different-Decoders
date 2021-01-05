import torch
import torch.nn as nn

import vocabulary
from encoder import EncoderCNN
from decoder import DecoderLSTM

####################################
### --- Baseline model ---
####################################

class BaselineConfig:
    # Force explicit overriding of fields by setting all fields to None.

    vocab_size = None
    use_glove = None
    embed_size = None

    encoder_dropout = None
    encoder_fine_tune = None

    decoder_hidden_size = None
    decoder_num_layers = None
    decoder_embed_dropout = None
    
class BaselineModel(nn.Module):
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        self.encoder = EncoderCNN(config.embed_size, dropout=config.encoder_dropout)
        if config.encoder_fine_tune: self.encoder.fine_tune(config.encoder_fine_tune)
        self.decoder = DecoderLSTM(config.embed_size, config.decoder_hidden_size, 
            config.vocab_size, config.decoder_num_layers, embed_dropout=config.decoder_embed_dropout, 
            use_glove=config.use_glove)

    def forward(self, imgs, captions):
        img_features = self.encoder(imgs)
        outputs = self.decoder(img_features, captions)
        return outputs

    def caption_image(self, img, vocab, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(img).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocab.i2w[predicted.item()] == vocabulary.END_TOKEN:
                    break

        return [vocab.i2w[idx] for idx in result_caption]