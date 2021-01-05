
import sys
import socket
import ssl
hostname = socket.gethostname()
if 'shannon' in hostname or 'markov' in hostname: 
    # Need this when downloading on university cluster...
    ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['NLTK_DATA'] = './nltk_data'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download('punkt')

from dataset import COCODataset
from model import BaselineModel, BaselineConfig
from vocabulary import PAD_TOKEN

def train_img_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(224)
        transforms.ToTensor(), 
        normalize])

def train_baseline_model(num_epochs=1, batch_size=32, num_workers=0, pin_memory=True):
    device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset.
    dataset = COCODataset(mode='train', img_transform=train_img_transform())

    # Create train data loader
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(batch):
        batch = tuple(zip(*batch))
        imgs, captions, caption_lengths = batch[0][:], batch[1][:], batch[2][:]
        imgs = torch.stack(imgs)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        caption_lengths = torch.stack(caption_lengths)
        return imgs, captions, caption_lengths
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned 
    # memory before returning them. If your data elements are a custom type,
    # or your collate_fn returns a batch that is a custom type.
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    vocab = dataset.vocab

    # Create model
    config = BaselineConfig()
    config.vocab_size = len(vocab)
    config.use_glove = False
    config.embed_size = 512
    config.encoder_dropout = 0
    config.encoder_fine_tune = False
    config.decoder_hidden_size = 512
    config.decoder_num_layers = 1
    config.decoder_embed_dropout = 0
    model = BaselineModel(config).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.w2i[PAD_TOKEN])
    learning_rate = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sys.stdout.flush()
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            imgs, captions, _ = batch

            imgs = imgs.to(device)
            captions = captions.to(device) # (batch_size, max_captions_length)

            # Remove the <end> token so that the model can learn to predict it.
            # We do this by first replacing all <end> tokens with <pad> token
            # and then remove the last element, which now practically is the <end> 
            # token.
            captions[captions==vocab('<end>')] = vocab('<pad>')
            captions = captions[:, :-1]

            outputs = model(imgs, captions) # (batch_size, max_captions_length, vocab_size)

            print(outputs.shape, captions.shape)

            #loss = criterion(outputs.contiguous().view(-1, config.vocab_size), captions.view(-1))
            break
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            print(loss)

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            if batch_idx == 5:
                break


if __name__ == '__main__':
    
    train_baseline_model(batch_size=16)
    
