import socket
import torch
import torch.nn as nn
import torchvision

def load_resnet101_model():
    hostname = socket.gethostname()
    if 'shannon' in hostname or 'markov' in hostname: 
        # Hack for working on the univeristy cluster. Pre-requisite:
        # "wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O resnet101.pth"
        model = torchvision.models.resnet101(pretrained=False)
        state_dict = torch.load('resnet101.pth')
        model.load_state_dict(state_dict)
    else:
        model = torchvision.models.resnet101(pretrained=True)
    return model

class EncoderCNN(nn.Module):
    """EncoderCNN

    Encoder for our image captioning task.
    """

    def __init__(self, embed_size, dropout=0):
        """Create layers for encoder.

        Args:
            embed_size (int): Correponds to the output dimension when doing forward propagation.
            dropout (float): Dropout probability. Default 0 (i.e. no dropout). 

        """

        super(EncoderCNN, self).__init__()

        # Load pre-trained ResNet101 model.
        resnet = load_resnet101_model()
        # Freeze pre-trained ResNet101 model.
        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)    
        
    def forward(self, imgs):
        """Forward propagation"""

        features = self.resnet(imgs)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.relu(features)
        features = self.dropout(features)
        return features

    def fine_tune(self, fine_tune=True):
        """Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        
        Args: 
            fine_tune: Whether to fine_tune or not.
        """

        # If fine-tuning is set, only fine-tune convolutional blocks 2 through 4
        for conv_block in list(self.resnet.children())[5:]:
            for param in conv_block.parameters():
                param.requires_grad = fine_tune