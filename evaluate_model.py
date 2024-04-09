import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision.transforms.functional as TF
from PIL import Image
import os
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor

from decoder import Decoder
from encoder import Encoder
from dataset import BlurSharpPairDataset

def tensor_to_PIL(tensor):
    tensor = tensor.squeeze(0)  # Remove the batch dimension
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    np_image = tensor.numpy()
    np_image = np.transpose(np_image, (1, 2, 0))  # Convert from PyTorch to PIL image format
    return Image.fromarray((np_image * 255).astype('uint8'))


if __name__ == '__main__':

    output_dir = 'deblurred_images'
    os.makedirs(output_dir, exist_ok=True)

    encoder = Encoder().to('cuda')
    decoder = Decoder().to('cuda')

    encoder.load_state_dict(torch.load('models/encoder_30.pt'))
    decoder.load_state_dict(torch.load('models/decoder_30.pt'))

    encoder.eval()
    decoder.eval()

    print("Models loaded")

# Function to convert a tensor to a PIL image

    # Example usage
    transform = Compose([
        Resize((256, 448)),
        ToTensor(),
    ])

    test_data = BlurSharpPairDataset(root_dir='data/test', transform=transform)

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    psnr_scores = []

    with torch.no_grad():
        for i, (blur_imgs, sharp_imgs) in enumerate(test_dataloader):
            encoded_features = encoder(blur_imgs.to('cuda'))
            deblurred_imgs = decoder(encoded_features.to('cuda'))

            # Save deblurred images
            save_image(deblurred_imgs, os.path.join(output_dir, f"deblurred_{i}.png"))

            # Convert images to the same type and range for PSNR calculation
            deblurred_imgs_np = deblurred_imgs.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            sharp_imgs_np = sharp_imgs.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

            psnr = compare_psnr(sharp_imgs_np, deblurred_imgs_np, data_range=255)
            psnr_scores.append(psnr)

    # Compute average PSNR
    average_psnr = np.mean(psnr_scores)
    print(f"Average PSNR: {average_psnr:.2f} dB")