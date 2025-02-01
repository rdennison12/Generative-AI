# Import the required libraries
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

"""
### **Summary**
This code is a classic implementation of a Generative Adversarial Network (GAN) for generating MNIST-like images. 
It trains a Generator to create fake images and simultaneously trains a Discriminator to distinguish real from fake images. 
Over time, the goal is for the Generator to produce images so realistic that the Discriminator cannot distinguish them 
from real data.
"""


# Visualization function
def show(tensor, ch=1, size=(28, 28), num=16):
    data = tensor.detach().cpu().view(-1, ch, *size)  # 128 x 1 x 28 x 28
    grid = make_grid(data[:num], nrow=4).permute(1, 2, 0)  # 1 x 28 x 28 = 28 x 28 x 1
    plt.imshow(grid)
    plt.show()


# setup of the main parameters and hyperparameters
epochs = 500
cur_step = 0
info_step = 300
mean_gen_loss = 0
mean_disc_loss = 0
z_dim = 64
lr = 0.00001
loss_func = nn.BCEWithLogitsLoss()
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=batch_size)


# declare our models

# Generator
def get_generator_block(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, z_dim=64, i_dim=784, h_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, h_dim),  # 64, 128
            get_generator_block(h_dim, h_dim * 2),  # 128, 256
            get_generator_block(h_dim * 2, h_dim * 4),  # 256 x 512
            get_generator_block(h_dim * 4, h_dim * 8),  # 512, 1024
            nn.Linear(h_dim * 8, i_dim),  # 1024, 784 (28x28)
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)


def generate_noise(number, z_dim):
    return torch.randn(number, z_dim).to(device)


# Discriminator
def get_discriminator_block(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(0.2, inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self, i_dim=784, h_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(i_dim, h_dim * 4),
            get_discriminator_block(h_dim * 4, h_dim * 2),
            get_discriminator_block(h_dim * 2, h_dim),
            nn.Linear(h_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)


gen = Generator(z_dim=z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# print(gen) These are required if using Colab, does not require the print function
# print(disc)

# Iterator
x, y = next(iter(dataloader))
print(x.shape, y.shape)
print(y[:10])

noise = generate_noise(batch_size, z_dim)
fake = gen(noise)
show(fake)


# Calculate the losses
# Generator loss
def calculate_gen_loss(loss_func, gen, disc, number, z_dim):
    noise = generate_noise(number, z_dim)
    fake = gen(noise)
    discriminator_prediction = disc(fake)
    gen_loss = loss_func(discriminator_prediction, torch.ones_like(discriminator_prediction))
    return gen_loss


# Discriminator loss
def calculate_disc_loss(loss_func, gen, disc, number, real, z_dim):
    noise = generate_noise(number, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    disc_fake_targets = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, disc_fake_targets)

    disc_real = disc(real)
    disc_real_targets = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_targets)

    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


# training loop
for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        disc_opt.zero_grad()
        cur_bs = len(real)
        real = real.view(cur_bs, -1).to(device)

        disc_loss = calculate_disc_loss(loss_func, gen, disc, cur_bs, real, z_dim)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = calculate_gen_loss(loss_func, gen, disc, cur_bs, z_dim)
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        # Visualization & Stats
        mean_disc_loss += disc_loss.item() / info_step
        mean_gen_loss += gen_loss.item() / info_step
        if cur_step % info_step == 0 and cur_step > 0:
            fake_noise = generate_noise(cur_bs, z_dim)
            fake = gen(fake_noise)
            show(fake)
            show(real)
            print(f"Epoch {epoch}, step {cur_step}/ : Generator loss: {mean_gen_loss} / discriminator loss: {mean_disc_loss}")
            mean_gen_loss = 0
            mean_disc_loss = 0
        cur_step += 1
