import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """CNN encoder."""

    def __init__(self, encoded_img_size=14, attention_method='ByPixel'):
        """Initialize encoder.

        Args:
            encoded_img_size (int): Output size.
            attention_method (str): Attention method to use. Supported attentions methods are "ByPixel" and "ByChannel".
        """

        assert attention_method in ['ByChannel', 'ByPixel']

        super(Encoder, self).__init__()

        self.enc_img_size = encoded_img_size
        self.att_method = attention_method

        # Load pre-trained ImageNet ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)

        # Remove linear and pool layers (top two layers) since we are not doing classification.
        # Specifically, remove: AdaptiveAvgPool2d(output_size=(1,1)) and Linear(in_features=2048, out_features=1000, bias=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        if self.att_method == "ByChannel":
            self.cnn1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU(inplace=True)

        # Resize image to fixed size to allow input images of variable size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_img_size, self.enc_img_size))

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
        if self.att_method == 'ByChannel': # (batch_size, encoder_dim, 8, 8) -> (batch_size, 512, 8, 8)
            features = self.relu(self.bn1(self.cnn1(features)))
        features = self.adaptive_pool(features) # (batch_size, 2048/512, 8, 8) -> (batch_size, encoder_dim/512, enc_img_size, enc_img_size)
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
