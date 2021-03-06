#this code is inspired by 
#https://github.com/sauravraghuvanshi/Udacity-Computer-Vision-Nanodegree-Program/blob/master/project_2_image_captioning_project/model.py
#https://github.com/ajamjoom/Image-Captions/blob/master/main.py
import socket
import torch
from torch import nn
import torchvision

def _load_resnet101_model():
    hostname = socket.gethostname()
    if 'shannon' in hostname or 'markov' in hostname:
        # Pre-requisite:
        # Run "wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth"
        # in the model directory.
        model = torchvision.models.resnet101(pretrained=False)
        state_dict = torch.load('models/resnet101.pth')
        model.load_state_dict(state_dict)
    else:
        model =  torchvision.models.resnet101(pretrained=True)
    return model

class Encoder(nn.Module):
    """CNN encoder."""

    def __init__(self, embed_size):
        """Initialize encoder.

        Args:
            encoded_img_size (int): Output size.
        """

        super(Encoder, self).__init__()
        
        # Cannot download resnet101 from torchvision due to certificate failure
        # on the cluster I'm running on. If you can, simply replace 
        # with resnet = torchvision.models.resnet101(pretrained=True)
        resnet = _load_resnet101_model()
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        """Forward propagation.

        Args:
            imgs (torch.Tensor): A tensor of dimension (batch_size, 3, img_size, img_size).

        Returns:
            Embedded image feature vectors of dimension (batch_size, embed_dim)
        """

        features = self.resnet(imgs)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

    def fine_tune(self, on=True):
        """Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Args:
            on (bool): Switch on or off.
        """

        for conv_block in list(self.resnet.children())[5:]:
            for param in conv_block.parameters():
                param.requires_grad = on


class EncoderAttention(nn.Module):
    """CNN encoder."""

    def __init__(self):
        """Initialize encoder.

        Args:
            encoded_img_size (int): Output size.
            attention_method (str): Attention method to use. Supported attentions methods are "ByPixel" and "ByChannel".
        """

        super(EncoderAttention, self).__init__()

        # Cannot download resnet101 from torchvision due to certificate failure
        # on the cluster I'm running on. If you can, simply replace 
        # with resnet = torchvision.models.resnet101(pretrained=True)
        resnet = _load_resnet101_model()

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        """Forward propagation.

        Args:
            imgs (torch.Tensor): A tensor of dimension (batch_size, 3, img_size, img_size).

        Returns:
            Encoded images of dimension (batch_size, encoded_img_size, encoded_img_size, 2048)
        """

        features = self.resnet(imgs)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        return features

    def fine_tune(self, on=True):
        """Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Args:
            on (bool): Switch on or off.
        """

        for conv_block in list(self.resnet.children())[5:]:
            for param in conv_block.parameters():
                param.requires_grad = on
