"""Background-conditioned foreground relighting with RelightDiffNet."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, CLIPTextModel, CLIPTokenizer

from pipeline_relight_diffnet import RelightDiffNetPipeline
from relight_diff_net import RelightingBranch
from unet_2d_condition import UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Relight a foreground with a target background.")
    parser.add_argument("--pretrained_model", type=str, required=True, help="Stable Diffusion v1.5 directory.")
    parser.add_argument("--relight_unet", type=str, required=True, help="Trained relighting branch directory.")
    parser.add_argument("--foreground", type=str, required=True)
    parser.add_argument("--background", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--seg_model", type=str, default=None, help="Optional RMBG model used when --mask is omitted.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_rgb(path, resolution):
    image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    array = np.asarray(image).astype(np.float32) / 255.0
    return array


def segment_foreground(seg_model, rgb, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pil_image = Image.fromarray((rgb * 255.0).astype(np.uint8))
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = seg_model(tensor)[-1].sigmoid()[0, 0].detach().cpu().numpy()
    return pred.astype(np.float32)


def main():
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    backbone = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    backbone.requires_grad_(False)
    backbone.eval()
    relight = RelightingBranch.from_pretrained(args.relight_unet)
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    pipeline = RelightDiffNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=backbone,
        relight_unet=relight,
        scheduler=scheduler,
    ).to(device)

    foreground = load_rgb(args.foreground, args.resolution)
    background = load_rgb(args.background, args.resolution)
    if args.mask is not None:
        mask = np.asarray(Image.open(args.mask).convert("L").resize((args.resolution, args.resolution), Image.NEAREST))
        mask = mask.astype(np.float32) / 255.0
    elif args.seg_model is not None:
        seg_model = AutoModelForImageSegmentation.from_pretrained(args.seg_model, trust_remote_code=True)
        seg_model.to(device).eval()
        mask = segment_foreground(seg_model, foreground, device)
    else:
        mask = np.ones(foreground.shape[:2], dtype=np.float32)

    masked_foreground = foreground * mask[..., None]
    masked_background = background * (1.0 - mask[..., None])
    image = np.concatenate([masked_foreground, masked_background], axis=2)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask)[None, None].to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipeline(
        prompt=args.prompt,
        image=image,
        mask=mask_tensor,
        background=image[:, 3:],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        blend=True,
    ).images[0]

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
