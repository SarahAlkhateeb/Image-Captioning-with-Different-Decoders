#this code is inspired by https://github.com/ajamjoom/Image-Captions/blob/master/main.py
def clip_gradient(optimizer, grad_clip):
    """Clips gradients computed during backpropagation to avoid explosion of gradients.

    Args:
        optimizer: Optimizer with the gradients to be clipped.
        grad_clip (int): Clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)