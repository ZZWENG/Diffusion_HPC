from typing import List, Optional, Union, Callable

import numpy as np
import torch
import PIL

from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from img_utils import randn_tensor, preprocess


class MyPipeline(StableDiffusionDepth2ImgPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        depth_map: Optional[torch.FloatTensor] = None,
        latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare depth mask
        depth_mask = self.prepare_depth_map(
            image,
            depth_map,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
            text_embeddings.dtype,
            device,
        )

        # 5. Preprocess image
        image = preprocess(image)
        # # import pdb; pdb.set_trace()
        # depth_mask[0, 0, 35:40, :30] += 0.3  # see how sensitive the model is to the depth map
        # depth_mask = torch.clamp(depth_mask, 0, 1)
        depth_array = ((depth_mask[0,0].cpu().numpy() + 1) * 127.5).astype(np.uint8)

        # 6. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        if latents is None:
            latents = self.prepare_latents(
                image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
            )  # (1, 4, 64, 64)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0]).type(torch.float16)
 
        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 10. Post-processing
        image = self.decode_latents(latents)

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image), depth_array

    def downsample_mask(self, mask_image, height, width, device):
        mask = [mask_image]
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        mask_downsample = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        ).to(device)
        return mask_downsample


def prepare_mask(mask_image):
    mask = [mask_image]
    mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
    mask = mask.astype(np.float32) / 255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    return mask

def get_random_latents(args, generator):
    latents = None
    seeds = []
    for _ in range(args.num_images):
        # Get a new random seed, store it and use it as the generator state
        seed = generator.seed()
        seeds.append(seed)
        generator = generator.manual_seed(seed)
        
        image_latents = torch.randn(
            (1, 4, args.height // 8, args.width // 8),
            generator = generator,
            device = device
        )
        latents = image_latents if latents is None else torch.cat((latents, image_latents))
        # latents should have shape (num_images, 4, 64, 64) in this case
    return latents

