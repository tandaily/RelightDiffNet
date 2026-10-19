import inspect
from typing import List, Optional, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

try:
    from models.relight_diff_net import predict_relight_noise
except ImportError:
    from relight_diff_net import predict_relight_noise


def _to_minus_one_to_one(image):
    if float(image.min()) >= 0:
        return image * 2.0 - 1.0
    return image


class RelightDiffNetPipeline(DiffusionPipeline):
    """DDIM sampling for the frozen generation branch conditioned by the relighting branch."""

    model_cpu_offload_seq = "text_encoder->relight_unet->unet->vae"
    _optional_components = []

    def __init__(self, vae, text_encoder, tokenizer, unet, relight_unet, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        object.__setattr__(self, "unet", unet)
        object.__setattr__(self, "relight_unet", relight_unet)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.unet.to(*args, **kwargs)
        self.relight_unet.to(*args, **kwargs)
        return self

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, num_images_per_prompt):
        if isinstance(prompt, str):
            prompt = [prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        if not do_classifier_free_guidance:
            return prompt_embeds

        uncond_inputs = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeds = self.text_encoder(uncond_inputs.input_ids.to(device))[0]
        uncond_embeds = uncond_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return torch.cat([uncond_embeds, prompt_embeds], dim=0)

    def _encode_image(self, image):
        image = _to_minus_one_to_one(image).to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(image).latent_dist.mode()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = "",
        image=None,
        foreground=None,
        background=None,
        mask=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
        blend: bool = True,
        generator: Optional[torch.Generator] = None,
        latents=None,
        output_type: str = "pil",
        return_dict: bool = True,
        eta: float = 0.0,
        **kwargs,
    ):
        if image is not None:
            if image.shape[1] < 6:
                raise ValueError("Expected `image` with foreground and background stacked on the channel axis.")
            foreground = image[:, :3]
            background = image[:, 3:6]
        if foreground is None or background is None:
            raise ValueError("Provide `image` or both `foreground` and `background`.")

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = foreground.shape[0]
        num_images_per_prompt = 1
        prompt_embeds = self._encode_prompt(prompt, device, do_classifier_free_guidance, num_images_per_prompt)

        fg_latents = self._encode_image(foreground)
        bg_latents = self._encode_image(background)
        if do_classifier_free_guidance:
            fg_latents = torch.cat([fg_latents, fg_latents], dim=0)
            bg_latents = torch.cat([bg_latents, bg_latents], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        height, width = foreground.shape[-2:]
        latent_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        else:
            latents = latents.to(device=device, dtype=prompt_embeds.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        step_signature = inspect.signature(self.scheduler.step).parameters
        extra_step_kwargs = {"eta": eta} if "eta" in step_signature else {}
        self.unet.eval()
        self.relight_unet.eval()

        for timestep in timesteps:
            latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = predict_relight_noise(
                self.unet,
                self.relight_unet,
                latent_model_input,
                timestep,
                prompt_embeds,
                fg_latents,
                bg_latents,
                conditioning_scale=conditioning_scale,
            )
            if do_classifier_free_guidance:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if blend and mask is not None:
            mask_tensor = mask
            if not torch.is_tensor(mask_tensor):
                mask_tensor = torch.as_tensor(mask_tensor)
            mask_tensor = mask_tensor.to(device=image.device, dtype=image.dtype)
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(1)
            if mask_tensor.shape[1] != 1 and mask_tensor.shape[1] != 3:
                mask_tensor = mask_tensor[:, :1]
            if mask_tensor.shape[-2:] != image.shape[-2:]:
                mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=image.shape[-2:], mode="nearest")
            background_image = _to_minus_one_to_one(background).to(device=image.device, dtype=image.dtype)
            if background_image.shape[-2:] != image.shape[-2:]:
                background_image = torch.nn.functional.interpolate(
                    background_image, size=image.shape[-2:], mode="bilinear", align_corners=False
                )
            image = mask_tensor * image + (1.0 - mask_tensor) * background_image

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image, None)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
