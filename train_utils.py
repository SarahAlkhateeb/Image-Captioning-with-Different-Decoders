

class AccumulatingMetric():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count

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