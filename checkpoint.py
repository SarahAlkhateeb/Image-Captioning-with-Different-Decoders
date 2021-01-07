
import os
import torch

CHECKPOINTS_DIR = 'checkpoints'


def load_checkpoint(device, args, verbose=True):
    """Loads model checkpoint.

    Returns:
        checkpoint: Checkpoint previously saved by calling save_checkpoint. 
    """

    path = os.path.join(CHECKPOINTS_DIR, f'{args.checkpoint}')
    if verbose:
        print(f'Loading checkpoint {path}')
    return torch.load(path, map_location=str(device))


def unpack_checkpoint(chkpt):
    """Unpacks a checkpoint.

    Args:
        checkpoint: Checkpoint previously saved by calling save_checkpoint.

    Returns:
        epoch (int): Epoch number.
        encoder (encoder.Encoder): Encoder model
        decoder: Decoder model
        encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
        decoder_optimizer: optimizer to update decoder's weights
        metrics (dict): Dictionary of metrics.
    """
    
    return chkpt['epoch'], chkpt['encoder'], chkpt['decoder'], chkpt['encoder_optimizer'], chkpt['decoder_optimizer'], chkpt['metrics']


def save_checkpoint(args, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, metrics, verbose=True):
    """Saves model checkpoint.

    Args:
        epoch (int): Epoch number.
        encoder (encoder.Encoder): Encoder model
        decoder: Decoder model
        encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
        decoder_optimizer: optimizer to update decoder's weights
        metrics (dict): Dictionary of metrics.
    """

    state = {
        'epoch': epoch,
        'metrics': metrics,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer,
    }
    path = os.path.join(CHECKPOINTS_DIR, f'{args.model_name}_{epoch}.pth.tar')
    torch.save(state, path)
    if verbose:
        print(f'Saved checkpoint to {path}')
