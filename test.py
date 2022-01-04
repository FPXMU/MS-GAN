import argparse
import torch
from models import Generator
from utils import save_images


def predict(args):
    generator = Generator().to(args.device)
    generator.load_state_dict(torch.load("./weights/generator.pth"))
    generator.eval()

    with torch.no_grad():
        latent_dim = min(args.base_channels * 2 ** args.num_stages, 512)
        z = torch.randn(args.batch_size, latent_dim, 1, 1, device=args.device)
        output = generator(z, 1, args.num_stages)
        output = output.permute(0, 2, 3, 1).cpu().numpy()
        save_images("./outputs/predict", output)


def main():
    parser = argparse.ArgumentParser(description="PGGAN")
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    predict(args)

if __name__ == '__main__':
    main()
