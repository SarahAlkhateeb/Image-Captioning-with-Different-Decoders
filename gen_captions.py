#this code is inpired by https://github.com/RoyalSkye/Image-Caption/blob/master/caption.py

import os
import sys
sys.path.append('cocoapi/PythonAPI/')
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from vocabulary import END_TOKEN, START_TOKEN, PAD_TOKEN, UNK_TOKEN, load_vocab
from checkpoint import load_checkpoint, unpack_checkpoint
import imageio
from PIL import Image
import numpy as np

def attention_caption_image_beam_search(device, args, img, encoder, decoder, vocab):
    """Reads an image and captions it with beam search.

    Note: Doesn't work for bert model sometimes doesn't converge...
     
    Args:
        device: Device to run on.
        args: Parsed command-line arguments from argparse.
        img (torch.Tensor): Image.
        encoder: Encoder model.
        decoder: Decoder model.
        vocab (vocabulary.Vocabulary): vocabulary

    Return: 
        caption, attention weights for visualization
    """

    k = args.beam_size
    Caption_End = False
    vocab_size = len(vocab)
    
    encoder_out = encoder(img)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # [1, num_pixels=196, encoder_dim]
    num_pixels = encoder_out.size(1)
    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[vocab(START_TOKEN)]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = torch.LongTensor([[vocab(START_TOKEN)]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)  # (s, 1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        cat_val = torch.cat([embeddings.double(), awe.double()], dim=1)
        h, c = decoder.decode_step(cat_val.float(), (h.float(), c.float()))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        
        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds]], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        print([vocab.i2w[id.item()] for id in next_word_inds])
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != vocab(END_TOKEN)]

        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            Caption_End = True
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    if not Caption_End:
        # If failure,
        return [vocab(START_TOKEN), vocab(END_TOKEN)], [], Caption_End
    else:
        indices = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[indices]
        alphas = complete_seqs_alpha[indices]

        return seq, alphas, Caption_End

def load_img(device, path):
    img = np.array(Image.fromarray(imageio.imread(path)))
    img = img.transpose(2, 0, 1)
    img = img / 255.0
    img = torch.FloatTensor(img).to(device)
    """ normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    img = transform(img)  # (3, 224, 224) """
    img = img.unsqueeze(0)  # (1, 3, 224, 224)
    return img
    
