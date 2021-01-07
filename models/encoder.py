import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """CNN encoder."""

    def __init__(self, embed_dim=None):
        """Initialize encoder.

        Args:
            encoded_img_size (int): Output size.
            attention_method (str): Attention method to use. Supported attentions methods are "ByPixel" and "ByChannel".
        """

        super(Encoder, self).__init__()

        # Load pre-trained ImageNet ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)

        if embed_dim is not None:
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            self.adaptive_pool = None
            self.embed = nn.Linear(resnet.fc.in_features, embed_dim)
        else:
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
            self.embed = None

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
        if self.embed is not None:
            features = features.view(features.size(0), -1)
            features = self.embed(features)
        else:
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
