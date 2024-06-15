from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter

from Discriminator import Discriminator
from Generator import Generator
from Save_model import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-4
batch_size = 128
image_size = 64
channels_img = 3
z_dim = 100
num_epoch = 2000
feature_Disc = 64
feature_Gen = 64
critic_iterations = 5

path = Path('Dataset')
transform = transforms.Compose(
      [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
      ]
)

class Dataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transforms = transform

    def __len__(self):
        return len(os.listdir(self.path))


    def __getitem__(self, index):
        dir = os.listdir(self.path)[index]
        img = path/f'{dir}'
        img = Image.open(img)
        img = self.transforms(img)
        return img

dataset = Dataset(path, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(z_dim, feature_Gen).to(device)
discriminator = Discriminator(channels_img,feature_Disc).to(device)


optimizer_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr,betas=(0.0, 0.9))


fixed_noise = torch.randn(batch_size,100, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

step = 0

generator.train()
discriminator.train()

for epoch in range(num_epoch):
    if(epoch%2 == 0 ):
        state = {
            "Generator": generator.state_dict(),
            "Discriminator": discriminator.state_dict(),
            "optimizer_gen": optimizer_gen.state_dict(),
            "optimizer_disc": optimizer_disc.state_dict(),
            "epoch": epoch,
        }
        save_model(state)

    for batch_idx, img in enumerate(dataloader):
        img = img.to(device)
        loss_disc = 0

        for _ in range(critic_iterations):
            noise = torch.randn(img.shape[0], 1, 1, 1).repeat(1,100,1,1).to(device)
            fake = generator(noise)
            epsilon = torch.randn(img.shape[0], 1, 1, 1).repeat(1, 3,image_size, image_size).to(device)
            x_hat = epsilon * img + (1 - epsilon) * fake
            x_hat_scores = discriminator(x_hat)
            gradient = torch.autograd.grad(inputs=x_hat, outputs=x_hat_scores, grad_outputs= torch.ones_like(x_hat_scores), create_graph=True, retain_graph=True)[0]
            gradient = gradient.view(gradient.shape[0], -1)
            gradient_norm = gradient.norm(2, dim=1)
            gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
            critic_real = discriminator(img).reshape(-1)
            critic_fake = discriminator(fake).reshape(-1)
            loss_critic = (-torch.mean(critic_real) + torch.mean(critic_fake) + 10*gradient_penalty)
            optimizer_disc.zero_grad()
            loss_critic.backward()
            optimizer_disc.step()
            loss_disc += loss_critic

        noise = torch.randn(img.shape[0], 1, 1, 1).repeat(1,100,1,1).to(device)
        fake = generator(noise)
        output = discriminator(fake).reshape(-1)
        loss_gen = -1*torch.mean(output)
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch}/{num_epoch}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_disc/5:.4f}, loss G: {loss_gen:.4f}")
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(img[:20], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:20], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1