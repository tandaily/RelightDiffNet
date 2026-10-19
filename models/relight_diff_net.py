"""Dual-branch relighting diffusion model.

The generation UNet stays frozen and only sees the noisy target latent. The
relighting UNet is a trainable copy. It sees the noisy latent concatenated
with the foreground and background latents, and its layer features are added
back into the generation UNet through zero-initialized convolutions.
"""

import copy
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.unet_2d_condition import UNet2DConditionModel
except ImportError:
    from unet_2d_condition import UNet2DConditionModel


def _block_out_channels(module):
    if hasattr(module, "out_channels") and module.out_channels is not None:
        return int(module.out_channels)
    return int(module.conv2.out_channels)


def zero_conv(channels):
    layer = nn.Conv2d(channels, channels, kernel_size=1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def expand_unet_input_channels(unet, extra_channels):
    """Widen conv_in. The original channels keep their pretrained weights."""
    if extra_channels <= 0:
        return unet
    old = unet.conv_in
    new = nn.Conv2d(
        old.in_channels + extra_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    )
    with torch.no_grad():
        new.weight.zero_()
        if new.bias is not None:
            new.bias.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=old.in_channels + extra_channels)
    return unet


def feature_channels(unet):
    """Channel counts of the features injected at each corresponding layer."""
    down_channels = [_block_out_channels(unet.conv_in)]
    for block in unet.down_blocks:
        for resnet in block.resnets:
            down_channels.append(_block_out_channels(resnet))
        downsamplers = getattr(block, "downsamplers", None)
        if downsamplers is not None and len(downsamplers) > 0:
            down_channels.append(_block_out_channels(block.resnets[-1]))

    mid_channels = _block_out_channels(unet.mid_block.resnets[-1])

    up_channels = []
    for block in unet.up_blocks:
        for resnet in block.resnets:
            up_channels.append(_block_out_channels(resnet))
    return down_channels, mid_channels, up_channels


class RelightingBranch(nn.Module):
    """Trainable relighting UNet plus the zero convolutions that feed the backbone."""

    def __init__(self, unet):
        super().__init__()
        if unet.config.in_channels != 12:
            raise ValueError(
                f"Relighting UNet expects 12 input channels (noisy + foreground + background), got {unet.config.in_channels}."
            )
        self.unet = unet
        down_channels, mid_channels, up_channels = feature_channels(unet)
        self.down_zero_convs = nn.ModuleList([zero_conv(channels) for channels in down_channels])
        self.mid_zero_conv = zero_conv(mid_channels)
        self.up_zero_convs = nn.ModuleList([zero_conv(channels) for channels in up_channels])

    @classmethod
    def from_unet(cls, unet, copy_weights=True):
        branch_unet = copy.deepcopy(unet) if copy_weights else unet
        if branch_unet.config.in_channels == 4:
            expand_unet_input_channels(branch_unet, extra_channels=8)
        return cls(branch_unet)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        torch.save(
            {
                "down": self.down_zero_convs.state_dict(),
                "mid": self.mid_zero_conv.state_dict(),
                "up": self.up_zero_convs.state_dict(),
            },
            os.path.join(save_directory, "zero_convs.pt"),
        )

    @classmethod
    def from_pretrained(cls, path):
        path = os.fspath(path)
        unet_dir = os.path.join(path, "unet")
        zero_path = os.path.join(path, "zero_convs.pt")
        if os.path.isdir(unet_dir):
            unet = UNet2DConditionModel.from_pretrained(unet_dir)
        else:
            unet = UNet2DConditionModel.from_pretrained(path)
            zero_path = ""
        if unet.config.in_channels == 4:
            expand_unet_input_channels(unet, extra_channels=8)
        branch = cls(unet)
        if zero_path and os.path.isfile(zero_path):
            state = torch.load(zero_path, map_location="cpu")
            branch.down_zero_convs.load_state_dict(state["down"])
            branch.mid_zero_conv.load_state_dict(state["mid"])
            branch.up_zero_convs.load_state_dict(state["up"])
        return branch

    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, cross_attention_kwargs=None):
        features = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_intermediates=True,
        )
        down_samples = features["down_block_res_samples"]
        mid_sample = features["mid_block_sample"]
        up_samples = features["up_block_res_samples"]
        if len(down_samples) != len(self.down_zero_convs) or len(up_samples) != len(self.up_zero_convs):
            raise RuntimeError(
                "Relighting features do not match zero convolutions: "
                f"down {len(down_samples)} vs {len(self.down_zero_convs)}, "
                f"up {len(up_samples)} vs {len(self.up_zero_convs)}."
            )

        down_residuals = tuple(conv(hidden) for conv, hidden in zip(self.down_zero_convs, down_samples))
        mid_residual = self.mid_zero_conv(mid_sample)
        up_residuals = tuple(conv(hidden) for conv, hidden in zip(self.up_zero_convs, up_samples))
        return down_residuals, mid_residual, up_residuals


def predict_relight_noise(
    unet,
    relight,
    noisy_latents,
    timesteps,
    encoder_hidden_states,
    fg_latents,
    bg_latents,
    class_labels=None,
    cross_attention_kwargs=None,
    conditioning_scale=1.0,
):
    """Run both branches and return the generation UNet's noise prediction."""
    relight_dtype = relight.unet.conv_in.weight.dtype
    backbone_dtype = unet.conv_in.weight.dtype
    fg_latents = fg_latents.to(dtype=relight_dtype)
    bg_latents = bg_latents.to(dtype=relight_dtype)
    relight_input = torch.cat([noisy_latents.to(dtype=relight_dtype), fg_latents, bg_latents], dim=1)
    down_residuals, mid_residual, up_residuals = relight(
        relight_input,
        timesteps,
        encoder_hidden_states.to(dtype=relight_dtype),
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    if conditioning_scale != 1.0:
        down_residuals = tuple(conditioning_scale * sample for sample in down_residuals)
        mid_residual = conditioning_scale * mid_residual
        up_residuals = tuple(conditioning_scale * sample for sample in up_residuals)
    down_residuals = tuple(sample.to(dtype=backbone_dtype) for sample in down_residuals)
    mid_residual = mid_residual.to(dtype=backbone_dtype)
    up_residuals = tuple(sample.to(dtype=backbone_dtype) for sample in up_residuals)

    return unet(
        noisy_latents.to(dtype=backbone_dtype),
        timesteps,
        encoder_hidden_states.to(dtype=backbone_dtype),
        class_labels=class_labels,
        cross_attention_kwargs=cross_attention_kwargs,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual,
        up_block_additional_residuals=up_residuals,
        return_dict=False,
    )[0]


class RelightDiffNet(nn.Module):
    def __init__(self, unet, relight):
        super().__init__()
        self.unet = unet
        self.relight = relight
        self.unet.requires_grad_(False)
        self.unet.eval()

    def train(self, mode=True):
        super().train(mode)
        self.unet.eval()
        return self

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        fg_latents,
        bg_latents,
        class_labels=None,
        cross_attention_kwargs=None,
        conditioning_scale=1.0,
    ):
        return predict_relight_noise(
            self.unet,
            self.relight,
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            fg_latents,
            bg_latents,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            conditioning_scale=conditioning_scale,
        )


def predict_x0(scheduler, noisy_latents, model_pred, timesteps):
    """One-step estimate of the clean latent x0 from the noise prediction."""
    alphas_cumprod = scheduler.alphas_cumprod.to(device=model_pred.device, dtype=torch.float32)
    alpha_bar = alphas_cumprod[timesteps.long()].view(-1, 1, 1, 1)
    noisy = noisy_latents.float()
    pred = model_pred.float()
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return (noisy - (1.0 - alpha_bar).sqrt() * pred) / alpha_bar.sqrt().clamp(min=1e-8)
    if prediction_type == "v_prediction":
        return alpha_bar.sqrt() * noisy - (1.0 - alpha_bar).sqrt() * pred
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


class PixelDomainLoss(nn.Module):
    """Masked spatial L1, official LPIPS, and time-adaptive frequency loss.

    Larger noise (larger t) puts more weight on the low-frequency error.
    Smaller t puts more weight on the high-frequency error.
    """

    def __init__(self, low_freq_radius=0.25):
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install the official LPIPS package with: pip install lpips") from exc
        self.lpips = lpips.LPIPS(net="alex")
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.low_freq_radius = low_freq_radius

    def train(self, mode=True):
        super().train(False)
        return self

    def _perceptual(self, pred, target):
        return self.lpips(pred.clamp(-1, 1), target.clamp(-1, 1)).mean()

    def _frequency(self, pred, target, timesteps, num_train_timesteps):
        pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho"), dim=(-2, -1))
        target_fft = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"), dim=(-2, -1))
        diff = (pred_fft.abs() - target_fft.abs()).abs()
        _, channels, height, width = diff.shape
        y_axis = torch.linspace(-1, 1, height, device=diff.device, dtype=diff.dtype)
        x_axis = torch.linspace(-1, 1, width, device=diff.device, dtype=diff.dtype)
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        low_mask = (radius <= self.low_freq_radius).to(dtype=diff.dtype)
        high_mask = 1.0 - low_mask
        low_denom = (low_mask.sum() * channels).clamp(min=1.0)
        high_denom = (high_mask.sum() * channels).clamp(min=1.0)
        low_loss = (diff * low_mask).sum(dim=(1, 2, 3)) / low_denom
        high_loss = (diff * high_mask).sum(dim=(1, 2, 3)) / high_denom
        low_weight = timesteps.float() / float(max(num_train_timesteps - 1, 1))
        return (low_weight * low_loss + (1.0 - low_weight) * high_loss).mean()

    def forward(self, pred, target, mask, timesteps, num_train_timesteps):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        if mask.shape[-2:] != pred.shape[-2:]:
            mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")

        autocast_ctx = (
            torch.autocast(device_type=pred.device.type, enabled=False) if pred.is_cuda else nullcontext()
        )
        with autocast_ctx:
            pred = pred.float()
            target = target.float()
            mask = mask.float()
            foreground = pred * mask + target * (1.0 - mask)
            spatial = ((pred - target).abs() * mask).sum() / (mask.sum() * pred.shape[1]).clamp(min=1.0)
            perceptual = self._perceptual(foreground, target)
            frequency = self._frequency(foreground, target, timesteps, num_train_timesteps)
        return {"l1": spatial, "lpips": perceptual, "freq": frequency}
