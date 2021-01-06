import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """CNN encoder."""

    def __init__(self):
        """Initialize encoder.

        Args:
            encoded_img_size (int): Output size.
            attention_method (str): Attention method to use. Supported attentions methods are "ByPixel" and "ByChannel".
        """

        super(Encoder, self).__init__()

        # Load pre-trained ImageNet ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers (top two layers) since we are not doing classification.
        # Specifically, remove: AdaptiveAvgPool2d(output_size=(1,1)) and Linear(in_features=2048, out_features=1000, bias=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size.
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

        features = self.resnet(imgs) # (batch_size, encoder_dim, img_size/32, img_size/32)
        features = self.adaptive_pool(features) # (batch_size, 2048/512, 8, 8) -> (batch_size, 2048/512, 14, 14)
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
