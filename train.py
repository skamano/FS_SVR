import os
import torch
import torchvision
from argparse import ArgumentParser
from torchvision import transforms
from torch.nn import BCELoss
from models import VAE, Flatten, UnFlatten


parser = ArgumentParser()
parser.add_argument("--epochs", help="Number of epochs for training on rendered RGB images from ShapeNet",
                    type=int, default=50)
parser.add_argument("--model", choices=["LEO", "VAE"], help="Specifies which architecture to train", 
                    default="VAE")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def train_VAE():
    vae = VAE(image_channels=3).to(device)

    if os.path.isfile('vae.torch'):
        vae.load_state_dict(torch.load('vae.torch', map_location='cpu'))
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        for idx, (images, _) in enumerate(dataloader):
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                    epochs, loss.data[0]/bs, bce.data[0]/bs, kld.data[0]/bs)
            print(to_print)

    torch.save(vae.state_dict(), 'vae.torch')


def train_LEO():
    pass


def main():
    if args.model == 'VAE':
        train_VAE()
    elif args.model == 'LEO':
        train_LEO()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()

