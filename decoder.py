import torch
import torch.nn as nn

class Decoder(nn.Module):
    
    def __init__(self, input_shape = (3, 256, 448), encoded_space_dim = 256):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 512),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(512, 128 * 31 * 55),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 31, 55))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=0, output_padding=1)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x