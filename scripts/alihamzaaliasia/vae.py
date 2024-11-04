import re
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from einops import einsum

load_pretrained = True



# Structure of the residual block
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1), # B, dim, H, W
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1,1, 0), # B, dim, H, W
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1), # B, dim, H/2, W/2
            nn.BatchNorm2d(dim), # B, dim, H/2, W/2
            nn.ReLU(True), # B, dim, H/2, W/2
            nn.Conv2d(dim, dim, 4, 2, 1), # B, dim, H/4, W/4
            ResBlock(dim), # B, dim, H/4, W/4
            ResBlock(dim), # B, dim, H/4, W/4
        )
        self.pre_codebook = nn.Conv2d(dim, dim, 1, 1, 0)
        self.codebook = nn.Embedding(K, dim)
        self.codebook.weight.data.uniform_(-1/K, 1/K)
        self.post_codebook = nn.Conv2d(dim, dim, 1, 1, 0)
        #incializa el codebook

        self.decoder = nn.Sequential(
            ResBlock(dim), # B, dim, H/4, W/4
            ResBlock(dim), # B, dim, H/4, W/4
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), # B, dim, H/2, W/2
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1), # B, input_dim, H, W
            nn.Tanh()
        )

    def quantize(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))

        dist = torch.cdist(x, self.codebook.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)

        #
        quant_out = torch.index_select(self.codebook.weight, 0, min_encoding_indices.view(-1))
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def forward(self, x):
        encoded_output = self.encoder(x)
        quant_input = self.pre_codebook(encoded_output)
        quantized, quantize_losses, min_encoding_indices = self.quantize(quant_input)
        dec_input = self.post_codebook(quantized)
        x_hat = self.decoder(dec_input)
        return x_hat, quantize_losses, min_encoding_indices



    def decode_from_codebook_indices(self, indices):
        quantized_output = self.quantize_indices(indices)
        dec_input = self.post_codebook(quantized_output)
        return self.decoder(dec_input)
    

def train(model, device, train_loader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, quantize_losses, min_encoding_indices = model(x)

            if epoch % 5 == 0 and i % 100 == 0:
                img1 = 255 * ((x[0].detach().permute(1, 2, 0).cpu().numpy() + 1) / 2)
                # Asegurar que los valores sean enteros de 8 bits                
                img1 = img1.astype(np.uint8)  
                # cv2.imwrite(f"output/generated/epoch_{epoch}_step_{i}_reconstructed.png", 255* ((x_hat[0].detach().permute(1, 2, 0).cpu().numpy()+1)/2))
                img2 = 255 * ((x_hat[0].detach().permute(1, 2, 0).cpu().numpy() + 1) / 2)
                img2 = img2.astype(np.uint8)
                cv2.imwrite(f"output/generated/epoch_{epoch}_step_{i}_combined.png",  np.vstack((img1, img2)))

            recon_loss = criterion(x_hat, x)
            loss = (
                1 * recon_loss +
                1 * quantize_losses['codebook_loss'] +
                0.2 * quantize_losses['commitment_loss']
            )
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}, Recon Loss: {recon_loss.item()}, Commitment Loss {quantize_losses['commitment_loss']} Codebook Loss {quantize_losses['codebook_loss']} ")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"output/vqvae_epoch_{epoch}_model.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformaciones para el dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Carga del dataset MNIST
    im_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


    train_loader = DataLoader(im_dataset, batch_size=32, shuffle=True)

    model = VectorQuantizedVAE(1, 32, 64)
    if load_pretrained:
        files = os.listdir("output")
        model_files = [f for f in files if f.startswith("vqvae_epoch_") and f.endswith("_model.pth")]
        epochs = [int(re.search(r"vqvae_epoch_(\d+)_model.pth", f).group(1)) for f in model_files]
        if epochs:  # If there are any model files
            max_epoch = max(epochs)
            model_file = f"output/vqvae_epoch_{max_epoch}_model.pth"
            # Load the model state
            model.load_state_dict(torch.load(model_file, map_location=device))
            print(f"Loaded pretrained model from {model_file}")
        else:
            print("No pretrained models found. Starting training from scratch.")


    train(model, device, train_loader, epochs=10)
    # validate(model, device, train_loader)


if __name__ == '__main__':
    main()