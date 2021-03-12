import argparse
import math
import os
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from dni_model.dni import GradualStyleEncoder
import lpips
from model import Generator


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def __load_latent_avg(ckpt, repeat=None):
    if 'latent_avg' in ckpt:
        latent_avg = ckpt['latent_avg'].to(device)
        if repeat is not None:
            latent_avg = latent_avg.repeat(repeat, 1)
    else:
        latent_avg = None
    return latent_avg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', help='Directory with images for encoding')
    parser.add_argument('--result_file', help='Directory for storing generated images')
    parser.add_argument('--dlatent_dir', help='Directory for storing dlatent representations')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--id_ckpt", type=str, default=None)
    parser.add_argument("--se_ckpt", type=str, default=None)
    parser.add_argument("--mode", type=str, default='DNI')
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--mse", type=float, default=0)
    parser.add_argument("--w_plus", action="store_true")

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    os.makedirs(args.result_file, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)
    os.makedirs('results/initial_images', exist_ok=True)

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    imgs = []

    for imgfile in ref_images:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=True)  # If false, then size=256 can fit it.
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True)

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    if args.mode == "DNI":
        id_ckpt = torch.load(args.id_ckpt, map_location=device)
        se_ckpt = torch.load(args.se_ckpt, map_location=device)

        encoder_identity = GradualStyleEncoder(50, 'ir_se').to(device)
        encoder_sematic = GradualStyleEncoder(50, 'ir_se').to(device)

        encoder_identity.load_state_dict(get_keys(id_ckpt, 'encoder'), strict=True)
        encoder_sematic.load_state_dict(se_ckpt['e'])

        encoder_identity.eval()
        encoder_sematic.eval()
        print('[DNI encoder loaded]')

        latent_avg = __load_latent_avg(id_ckpt)

        codes_identy = encoder_identity(imgs.float())
        codes_sematic = encoder_sematic(imgs.float())

        latent_in = codes_identy + latent_avg.repeat(codes_identy.shape[0], 1, 1) + 1e-5 * codes_sematic

        # print(latent_in.shape)

        initial_images, _ = g_ema([latent_in],
                                  input_is_latent=True,
                                  randomize_noise=False,
                                  return_latents=False)

        img_initial = make_image(initial_images)

        for i, input_name in enumerate(ref_images):
            img_name_initial = os.path.splitext(os.path.basename(input_name))[0] + '-initial'
            pil_img_ini = Image.fromarray(img_initial[i])
            pil_img_ini.save(os.path.join('results/initial_images', f'{img_name_initial}.png'), 'PNG')

        print("initial code from DNI")

        latent_in = latent_in.detach().clone()

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        # print(latent_n.shape) # 1, 18, 512
        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])  # reshape 1024 -> 256

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    filename = os.path.join(args.result_file, os.path.splitext(os.path.basename(ref_images[0]))[0] + ".pt")

    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(ref_images):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i: i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
        }
        img_name = os.path.splitext(os.path.basename(input_name))[0]
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(os.path.join(args.result_file, f'{img_name}.png'), 'PNG')
        np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), latent_in[i].detach().cpu().numpy())

    torch.save(result_file, filename)
