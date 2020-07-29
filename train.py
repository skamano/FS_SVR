import os
import torch
import torchvision
import torch.nn.functional as F
from argparse import ArgumentParser
from torchvision import transforms
from torch.nn import BCELoss, init
from models import VAE, Flatten, UnFlatten
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from datasets import ShapeNetDataset


parser = ArgumentParser()
parser.add_argument("--epochs", help="Number of epochs for training on rendered RGB images from ShapeNet",
                    type=int, default=500)
parser.add_argument("--model", choices=["LEO", "VAE"], help="Specifies which architecture to train", 
                    default="VAE")
parser.add_argument("--batch_size", choices=[4, 8, 16, 32, 64], help="Batch size for training.",
                    default=64)
parser.add_argument("--num_workers", choices=list(range(1, 11)), 
                    help="Number of worker threads for batched dataloading.",
                    default=4)
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
    transform = transforms.Compose([transforms.CenterCrop(127),
                                   transforms.ToTensor()])

    print("Loading ShapeNet...")
    shapenet = ShapeNetDataset(root_dir='/home/svcl-oowl/dataset/shapenet',
                               transform=transform)

    print("Initializing dataloader: {} workers allocated".format(args.num_workers))
    dataloader = DataLoader(shapenet, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers, 
                            worker_init_fn=shapenet.worker_init_fn)

    print("Initializing model...")
    vae = VAE(image_channels=3).to(device)
    print(vae)

    if os.path.isfile('vae.torch'):
        print("Previous checkpoint found. Loading from memory...")
        vae.load_state_dict(torch.load('vae.torch', map_location=device))
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        for i_batch, images in enumerate(dataloader):
            images = images.to(device)
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = args.batch_size
            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                    args.epochs, loss.data.item()/bs, bce.data.item()/bs, kld.data.item()/bs)
            print(to_print)
            print("Images processed[{}/{}]".format(i_batch*bs, shapenet.length))

        torch.save(vae.state_dict(), 'vae.torch')


def train_LEO():
    raise NotImplementedError


def main():
    if args.model == 'VAE':
        print("Training VAE")
        train_VAE()
    elif args.model == 'LEO':
        train_LEO()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()

