import gc
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..config import SHOT_NUM_CONDITION_FRAMES_MAP
from ..modules import get_text_encoder, get_transformer, get_vae
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.util import get_video_info


class ShotSwitchingExtensionPipeline:
    """Cinematic **shot-switching** extension (report §2.2), up to 5 s per run.

    Loads ``shot_transformer`` (not the single-shot ``transformer``). The number of
    tail frames used as VAE context is **tied to** ``--duration`` via
    ``SHOT_NUM_CONDITION_FRAMES_MAP`` in ``config`` (2–5 s). Shot-type keywords such as
    ``[ZOOM_IN_CUT]`` are **not** parsed here; they are prompt conventions / LLM hints
    for the user. The training-time shot detector described in the paper is out of scope
    for this inference repo.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=False,
        low_vram=False,
    ):
        """
        Initialize the diffusion forcing pipeline class

        Args:
            model_path (str): Path to the model
            device (str): Device to run on, defaults to 'cuda'
            weight_dtype: Weight data type, defaults to torch.bfloat16
        """
        offload = offload or low_vram
        load_device = "cpu" if offload else device
        self.transformer = get_transformer(
            model_path,
            subfolder="shot_transformer",
            device=load_device,
            weight_dtype=weight_dtype,
            low_vram=low_vram,
        )
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, device=device, weight_dtype=torch.float32)
        self.text_encoder = get_text_encoder(
            model_path, device=load_device, weight_dtype=weight_dtype
        )
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.device = device
        self.offload = offload
        self.low_vram = low_vram
        self.sp_size = 1

        if use_usp:
            import types

            from xfuser.core.distributed import get_sequence_parallel_world_size

            from ..distributed.context_parallel_for_extension import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in self.transformer.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
                self.transformer.forward = types.MethodType(
                    usp_dit_forward, self.transformer
                )
                self.sp_size = get_sequence_parallel_world_size()

        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.components = {
            "vae": self.vae.vae,
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
        }
        self.offload = offload
        self.vae.to(self.device)
        if self.offload:
            self.text_encoder.to("cpu")
            self.transformer.to("cpu")
        else:
            self.text_encoder.to(self.device)
            self.transformer.to(self.device)

    def extend_video(
        self,
        raw_video: str,
        prompt: str,
        duration: int,
        seed: int,
        fps: int = 24,
        resolution: str = "720P",
    ):
        """One-shot extension with duration-dependent conditioning depth.

        Pseudocode::

            assert duration in SHOT_NUM_CONDITION_FRAMES_MAP
            N <- map[duration]
            prefix <- last N frames  # ``get_video_info``
            cond <- VAE.encode(prefix)
            return denoise(cond, num_frames=duration*fps+1, prompt)

        Unlike ``SingleShotExtensionPipeline``, there is **no** rolling loop — full
        output must fit the shot-switching model's trained horizon.
        """
        assert (
            duration in SHOT_NUM_CONDITION_FRAMES_MAP
        ), f"Duration {duration} not supported"
        num_condition_frames = SHOT_NUM_CONDITION_FRAMES_MAP[duration]
        frames_num = duration * fps + 1
        prefix_video, raw_video, height, width = get_video_info(
            raw_video, num_condition_frames, resolution
        )
        prefix_video = prefix_video.to(self.device)
        prefix_video = self.vae.encode(prefix_video)
        video_frames = self.__call__(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_frames=frames_num,
            num_inference_steps=8,
            guidance_scale=1.0,
            shift=8.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            prefix_video=prefix_video,
            block_offload=self.low_vram,
        )[0]
        logging.info(f"video_frames: {video_frames.shape}")
        return video_frames

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        width: int = 544,
        height: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        generator: Optional[torch.Generator] = None,
        block_offload: bool = False,
        **kwargs,
    ):
        self._guidance_scale = guidance_scale
        if self.offload:
            self.text_encoder.to(self.device)
        # preprocess
        F = num_frames
        target_shape = (
            self.vae.vae.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = (
            self.text_encoder.encode(negative_prompt).to(self.device)
            if self.do_classifier_free_guidance
            else None
        )
        if self.offload:
            self.text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        latents = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
        ]

        prefix_video = kwargs["prefix_video"].to(self.device)

        context_frames = prefix_video.shape[2]
        total_frames = latents[0].shape[1] + context_frames

        if self.offload and not block_offload:
            self.transformer.to(self.device)
        else:
            self.transformer.to("cpu")
        with torch.cuda.amp.autocast(dtype=self.transformer.dtype), torch.no_grad():
            self.scheduler.set_timesteps(
                num_inference_steps, device=self.device, shift=shift
            )
            timesteps = self.scheduler.timesteps

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = torch.stack(latents)
                timestep = t.repeat(latent_model_input.shape[0])
                timestep = timestep.unsqueeze(-1).repeat(1, total_frames)
                timestep[:, -context_frames:] = 0
                if guidance_scale > 1.0:
                    noise_pred_cond = self.transformer(
                        [latent_model_input, prefix_video], t=timestep, context=context, block_offload=block_offload
                    )[0]
                    noise_pred_uncond = self.transformer(
                        [latent_model_input, prefix_video],
                        t=timestep,
                        context=context_null,
                        block_offload=block_offload,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = self.transformer(
                        [latent_model_input, prefix_video], t=timestep, context=context, block_offload=block_offload
                    )
                    # Check if the output is a tuple or a tensor
                    if isinstance(noise_pred, tuple):
                        noise_pred = noise_pred[0]

                temp_x0 = self.scheduler.step(
                    noise_pred,
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                latents = [temp_x0.squeeze(0)]
                if block_offload:
                    gc.collect()
                    torch.cuda.empty_cache()
            if self.offload:
                self.transformer.cpu()
                torch.cuda.empty_cache()
            videos = self.vae.decode(latents[0])
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video for video in videos]
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
        if self.offload:
            gc.collect()
            torch.cuda.empty_cache()
        return videos
