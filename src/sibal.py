import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils


from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [file for file in os.listdir(image_dir) if file.lower().endswith('.png') or file.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image


class CustomTestDataset(Dataset):
    def __init__(self, image_dir,mask_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [file for file in os.listdir(image_dir) if file.lower().endswith('.png') or file.lower().endswith('.jpg')]
        self.mask_dir= mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join( self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert('L')  # convert image to grayscale
        mask = Image.open(mask_path).convert('L')  # convert image to grayscale
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image,mask
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # Output: (32, 128, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # Output: (64, 64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # Output: (128, 32, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.enc4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # Output: (128, 32, 32)
        self.bn4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dec1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # Output: (64, 64, 64)
        self.bn4 = nn.BatchNorm2d(64)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # Output: (32, 128, 128)
        self.bn5 = nn.BatchNorm2d(32)
        self.dec3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # Output: (1, 256, 256)

    def forward(self, x):
        # Encoder
        x = F.leaky_relu(self.bn1(self.enc1(x)))
        x = F.leaky_relu(self.bn2(self.enc2(x)))
        x = F.leaky_relu(self.bn3(self.enc3(x)))

        # Decoder
        x = F.leaky_relu(self.bn4(self.dec1(x)))
        x = F.leaky_relu(self.bn5(self.dec2(x)))
        x = torch.sigmoid(self.dec3(x))

        return x

# class Encoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Encoder, self).__init__()
#         # Adjust the number of layers or kernel sizes if necessary
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 128x128
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 64x64
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 32x32
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 16x16
#         self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
#         self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = torch.flatten(x, start_dim=1)
#         return self.fc_mu(x), self.fc_logvar(x)

# class Decoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
#         self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # Output: 1x256x256

#     def forward(self, z):
#         z = self.fc(z)
#         z = z.view(-1, 256, 16, 16)  # Reshape to match the spatial dimensions
#         z = F.relu(self.deconv1(z))
#         z = F.relu(self.deconv2(z))
#         z = F.relu(self.deconv3(z))
#         return torch.sigmoid(self.deconv4(z))

# class GMVAE(nn.Module):
#     def __init__(self, latent_dim):
#         super(GMVAE, self).__init__()
#         self.encoder = Encoder(latent_dim)
#         self.decoder = Decoder(latent_dim)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar
    
criterion = nn.MSELoss()
    
def loss_function(recon_x, x, mu, logvar):
    BCE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust as per your dataset's requirement
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
        

train_dataset = CustomDataset(image_dir='./image_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

small_train_dataset = CustomDataset(image_dir='./small', transform=transform)
small_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# testing
test_dataset = CustomTestDataset(image_dir='./ano_image_data',mask_dir='./mask', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

latent_dim = 128
#model = GMVAE(latent_dim)
model = ConvAutoencoder().to(device)

lr_decay_factor = 0.5  # Learning rate decay factor
lr_decay_step = 10 
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)


def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )


def heatmap(real: torch.Tensor, recon: torch.Tensor, mask:torch.Tensor, filename=None, save=True):
    mse = ((recon.detach().cpu() - real.detach().cpu()).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    if save:
        output = torch.cat((real.detach().cpu(), recon.detach().cpu(), mse, mse_threshold,mask))
        plt.figure(figsize=(12, 5)) 
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        if filename != None:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

import numpy as np
import matplotlib.pyplot as plt

def compare_images_batch(original_batch, reconstructed_batch,filename=None):
    """
    Compare batches of original and reconstructed images (as torch tensors),
    calculate L2 reconstruction error for each pair, and plot the errors on the image comparisons.

    Parameters:
    original_batch (torch.Tensor): Batch of original images.
    reconstructed_batch (torch.Tensor): Batch of reconstructed images.
    """

    batch_size = original_batch.size(0)
    for i in range(batch_size):
        # Convert tensors to NumPy arrays
        original = original_batch[i].detach().cpu().numpy()
        reconstructed = reconstructed_batch[i].detach().cpu().numpy()

        # Squeeze the single channel dimension if it's a grayscale image
        if original.shape[0] == 1:
            original = np.squeeze(original, axis=0)
            reconstructed = np.squeeze(reconstructed, axis=0)
        else:  # For RGB images, transpose the dimensions
            original = np.transpose(original, (1, 2, 0))
            reconstructed = np.transpose(reconstructed, (1, 2, 0))

        # Normalize the image data to 0-1 for displaying
        original = (original - original.min()) / (original.max() - original.min())
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

        # Calculate L2 norm (Euclidean distance) as the reconstruction error for each image
        error = np.linalg.norm(original - reconstructed)

        # Plotting the original and reconstructed images
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(reconstructed, cmap='gray')
        axes[1].set_title(f'Reconstructed Image\nReconstruction Error (L2 Norm): {error:.2f}')
        axes[1].axis('off')

        plt.tight_layout()
        if filename!=None:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

num_epochs = 100
import pdb
model.train()
for epoch in range(num_epochs):
    for tt,data in enumerate(train_loader):
        imgs = data.to(device)
        #recon_batch, mu, logvar = model(imgs)
        recon_batch = model(imgs)
        #loss = loss_function(recon_batch, imgs, mu, logvar)
        loss = criterion(recon_batch, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch+1},batch-:{tt+1}, Loss:{loss.item():.4f}')
    if epoch%5==0:
        compare_images_batch(imgs,recon_batch,filename='./results/comp/comp_%d_%d.png'%(epoch,tt))
        scheduler.step()
    if epoch%10==0:
        print('Testing results -epoch %d'%(epoch))
        for t,tdata in enumerate(test_loader):
            timg,tmask = tdata
            trecon_batch = model(timg.to(device))
            _len = len(timg)
            for i in range(_len):
                heatmap(timg[i:i+1],trecon_batch[i:i+1],tmask[i:i+1],filename='./results/det/eval_det_%d_%d_%d.png'%(epoch,t,i))
            if t==5:
                break
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')

torch.save(model.state_dict(), 'conv_autoencoder.pth')
#Visualisation of training datsaet
for _num_f,data in enumerate(small_train_loader):
    imgs = data.to(device)
    #recon_batch, mu, logvar = model(imgs)
    recon_batch = model(imgs.to(device))
    compare_images_batch(imgs,recon_batch,filename='./results/comp1/final_small_comp_%d.png'%(_num_f))
    

#Visualisation of test dataset
for tdata in test_loader:
    timg,tmask = tdata
    trecon_batch = model(timg.to(device))
    compare_images_batch(timg,trecon_batch)
    _len = len(timg)
    for i in range(_len):
        heatmap(timg[i:i+1],trecon_batch[i:i+1],tmask[i:i+1],filename='./results/det1/final_test_det_%d_%d_%d.png'%(epoch,t,i))
    

