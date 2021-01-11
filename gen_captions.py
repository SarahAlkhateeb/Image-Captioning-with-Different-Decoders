import time
import os
import sys
sys.path.append('cocoapi/PythonAPI/')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from checkpoint import load_checkpoint, unpack_checkpoint
from metric import AccumulatingMetric
from dataset import COCODataset
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from vocabulary import END_TOKEN, PAD_TOKEN, START_TOKEN, Vocabulary

import os, time
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

from metric import get_eval_score

def attention_caption_image_beam_search(device, args, img, encoder, decoder, vocab):
    """Reads an image and captions it with beam search.
     
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
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
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

    assert Caption_End
    indices = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[indices]
    alphas = complete_seqs_alpha[indices]

    return seq, alphas



def evaluate(device, args, encoder, decoder):
    """Performs one epoch's evaluation.

    Args:
        val_loader: DataLoader for validation data.
        encoder: Encoder model
        Decoder: Decoder model
        criterion: Loss layer
    
    Returns:
        score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    dataset = COCODataset(
        mode='val', img_transform=img_transform, caption_max_len=args.max_caption_length)

    vocab = dataset.vocab

    # Dataloader.
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    decoder.eval()
    encoder.eval()

    batch_time = AccumulatingMetric()
    losses = AccumulatingMetric()
    top5accs = AccumulatingMetric()

    start = time.time()

    references = []  # Eeferences (true captions) for calculating BLEU-4 score
    hypotheses = []  # Hypotheses (predictions)

    # Explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():

        # Batches
        for batch_idx, (img, caption, img_path, all_captions) in enumerate(val_loader):    
            img = img.to(device)

            seq, _ = attention_caption_image_beam_search(device, args, img, encoder, decoder, vocab)
         
            img_captions = list(
                map(lambda c: [w for w in c if w not in {vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN)}],
                    all_captions[0].tolist()))  
            references.append(img_captions)

            hypotheses.append([w for w in seq if w not in {vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN)}])
            assert len(references) == len(hypotheses)

            break
            
    metrics = get_eval_score(references, hypotheses)
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Generate caption')
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint of trained model.')
    parser.add_argument('--max_caption_length', type=int, default=-1,
                        help='only use captions with caption length <= 50 when training.')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size for beam search')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    chkpt = load_checkpoint(device, args)
    _, encoder, decoder, _, _, _ = unpack_checkpoint(chkpt)

    evaluate(device, args, encoder, decoder)

if __name__ == '__main__':
    main()
