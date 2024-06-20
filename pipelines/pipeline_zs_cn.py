# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import PIL.Image

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.utils import is_accelerate_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers import DiffusionPipeline, UNet3DConditionModel, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class FusionEditPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder_v: CLIPTextModel,
        text_encoder_i: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet_v: UNet3DConditionModel,
        unet_i: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        # feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__()

        # if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
        #     deprecation_message = (
        #         f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
        #         f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
        #         "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
        #         " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
        #         " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
        #         " file"
        #     )
        #     deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
        #     new_config = dict(scheduler.config)
        #     new_config["steps_offset"] = 1
        #     scheduler._internal_dict = FrozenDict(new_config)

        # if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
        #     deprecation_message = (
        #         f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
        #         " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
        #         " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
        #         " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
        #         " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
        #     )
        #     deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
        #     new_config = dict(scheduler.config)
        #     new_config["clip_sample"] = False
        #     scheduler._internal_dict = FrozenDict(new_config)

        # is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
        #     version.parse(unet.config._diffusers_version).base_version
        # ) < version.parse("0.9.0.dev0")
        # is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        # if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
        #     deprecation_message = (
        #         "The configuration file of the unet has set the default `sample_size` to smaller than"
        #         " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
        #         " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
        #         " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
        #         " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
        #         " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
        #         " in the config might lead to incorrect results in future versions. If you have downloaded this"
        #         " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
        #         " the `unet/config.json` file"
        #     )
        #     deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
        #     new_config = dict(unet.config)
        #     new_config["sample_size"] = 64
        #     unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder_v=text_encoder_v,
            text_encoder_i=text_encoder_i,
            tokenizer=tokenizer,
            unet_i=unet_i,
            unet_v=unet_v,
            controlnet=controlnet,
            scheduler=scheduler,
            # feature_extractor=feature_extractor,
            # image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder_v, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, flag):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if flag == 'video':
            text_encoder = self.text_encoder_v
        elif flag == 'image':
            text_encoder = self.text_encoder_i

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def check_inputs(
        self,
        prompt,
        # image,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        
        # # check image
        # is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
        #     self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        # )
        # if (
        #     isinstance(self.controlnet, ControlNetModel)
        #     or is_compiled
        #     and isinstance(self.controlnet._orig_mod, ControlNetModel)
        # ):
        #     self.check_image(image, prompt, prompt_embeds)
        # elif (
        #     isinstance(self.controlnet, MultiControlNetModel)
        #     or is_compiled
        #     and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        # ):
        #     if not isinstance(image, list):
        #         raise TypeError("For multiple controlnets: `image` must be type `list`")

        #     # When `image` is a nested list:
        #     # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
        #     elif any(isinstance(i, list) for i in image):
        #         transposed_image = [list(t) for t in zip(*image)]
        #         if len(transposed_image) != len(self.controlnet.nets):
        #             raise ValueError(
        #                 f"For multiple controlnets: if you pass`image` as a list of list, each sublist must have the same length as the number of controlnets, but the sublists in `image` got {len(transposed_image)} images and {len(self.controlnet.nets)} ControlNets."
        #             )
        #         for image_ in transposed_image:
        #             self.check_image(image_, prompt, prompt_embeds)
        #     elif len(image) != len(self.controlnet.nets):
        #         raise ValueError(
        #             f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
        #         )
        #     else:
        #         for image_ in image:
        #             self.check_image(image_, prompt, prompt_embeds)
        # else:
        #     assert False


    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        # if do_classifier_free_guidance and not guess_mode:
        #     image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt_i: Union[str, List[str]] = None,
        prompt_v: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        video_length: Optional[int] = 16,
        fusion_alpha_v: float = 0.5,
        fusion_alpha_i: float = 0.5,
        decay: bool = True,
        reverse: bool = False,
        denoise_time: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale_v: float = 7.5,
        guidance_scale_i: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents_v: Optional[torch.FloatTensor] = None,
        latents_i: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            # mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            mult = 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # Check inputs. Raise error if not correct
        self.check_inputs(
            prompt_v,
            # image,
            height,
            width,
            callback_steps,
            negative_prompt,
            control_guidance_start,
            control_guidance_end,
        )


        # Define call parameters
        batch_size = 1 if isinstance(prompt_v, str) else len(prompt_v)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance_v = guidance_scale_v > 1.0
        do_classifier_free_guidance_i = guidance_scale_i > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # Encode input prompt
        text_embeddings_v = self._encode_prompt(
            prompt_v, device, num_videos_per_prompt, do_classifier_free_guidance_v, negative_prompt, flag='video'
        )

        text_embeddings_i = self._encode_prompt(
            prompt_i, device, num_images_per_prompt, do_classifier_free_guidance_v, negative_prompt, flag='image'
        )

        # prepare image
        images = []
        if isinstance(controlnet, ControlNetModel):
            # for image_ in image:
            #     image_ = self.prepare_image(
            #         image=image_,
            #         width=width,
            #         height=height,
            #         batch_size=batch_size * num_images_per_prompt,
            #         num_images_per_prompt=num_images_per_prompt,
            #         device=device,
            #         dtype=controlnet.dtype,
            #         do_classifier_free_guidance=do_classifier_free_guidance_i,
            #         guess_mode=guess_mode,
            #     )
            #     images.append(image_)
            # height, width = images[0].shape[-2:]
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance_i,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            # Nested lists as ControlNet condition
            if isinstance(image[0], list):
                # Transpose the nested image list
                image = [list(t) for t in zip(*image)]

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance_i,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # duplicate image for video_length
        images = []
        for j in range(image.shape[0]):
            images.append(rearrange(image[j], " c h w -> 1 c h w"))


        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet_v.in_channels
        latents_v = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings_v.dtype,
            device,
            generator,
            latents_v,
        )
        latents_dtype = latents_v.dtype
        num_channels_latents = self.unet_i.in_channels
        latents_i = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings_i.dtype,
            device,
            generator,
            latents_i,
        )
        latents_is = []
        for latent in torch.chunk(latents_i, video_length, dim=2):
            latents_is.append(torch.squeeze(latent, dim=2))

        # Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # set alpha


        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        delta_fusion_alpha = 1 - fusion_alpha_v

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # denoise of vdm
                # expand the latents of vdm if we are doing classifier free guidance
                latent_model_input_v = torch.cat([latents_v] * 2) if do_classifier_free_guidance_v else latents_v
                latent_model_input_v = self.scheduler.scale_model_input(latent_model_input_v, t)

                # predict the noise residual of vdm
                # print(latent_model_input_v.shape)
                noise_pred_v = self.unet_v(latent_model_input_v, t, encoder_hidden_states=text_embeddings_v).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance_v:
                    noise_pred_uncond_v, noise_pred_text_v = noise_pred_v.chunk(2)
                    noise_pred_v = noise_pred_uncond_v + guidance_scale_v * (noise_pred_text_v - noise_pred_uncond_v)

                # reshape latents_v
                bsz, channel, frames, width, height = latents_v.shape
                latents_v = latents_v.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred_v = noise_pred_v.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents_v = self.scheduler.step(noise_pred_v, t, latents_v, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents_v = latents_v[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # denoise of idm
                for f in range(video_length):
                    x = latents_is[f]
                    latent_model_input_i = torch.cat([x] * 2) if do_classifier_free_guidance_i else x
                    latent_model_input_i = self.scheduler.scale_model_input(latent_model_input_i, t)

                    if guess_mode and self.do_classifier_free_guidance_i:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = x
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = text_embeddings_i.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input_i
                        controlnet_prompt_embeds = text_embeddings_i

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    # print(control_model_input.shape)
                    # print(torch.cat([images[f]] * 2).shape)
                    # exit(0)


                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=torch.cat([images[f]] * 2),
                        # controlnet_cond=images[f].tensor(),
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and self.do_classifier_free_guidance_i:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                    

                    noise_pred_i = self.unet_i(
                        latent_model_input_i,
                        t,
                        encoder_hidden_states=text_embeddings_i,
                        # timestep_cond=timestep_cond,
                        # cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        # added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance_i:
                        noise_pred_uncond_i, noise_pred_text_i = noise_pred_i.chunk(2)
                        noise_pred_i = noise_pred_uncond_i + guidance_scale_i * (noise_pred_text_i - noise_pred_uncond_i)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_i = self.scheduler.step(noise_pred_i, t, x, **extra_step_kwargs, return_dict=False)[0]

                    latents_is[f] = latents_i

                # latent fusion
                if i >= denoise_time:
                    for index in range(len(latents_is)):
                        latents_is[index] = rearrange(latents_is[index], "b c h w -> b c 1 h w")
                    latents_i = torch.cat(latents_is, dim=2)

                    latents = fusion_alpha_v * latents_v + fusion_alpha_i * latents_i
                    if decay:
                        if not reverse:
                            alpha = fusion_alpha_v + delta_fusion_alpha / (len(timesteps) - denoise_time)
                            fusion_alpha_v = alpha if alpha <= 1 else 1
                        else:
                            alpha = fusion_alpha_v - (1 - delta_fusion_alpha) / (len(timesteps) - denoise_time)
                            fusion_alpha_v = alpha if alpha > 0 else 0
                        fusion_alpha_i = 1 - fusion_alpha_v
                    latents_v = latents
                    latents_is = []
                    for latent in torch.chunk(latents, video_length, dim=2):
                        latents_is.append(torch.squeeze(latent, dim=2))

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # # Post-processing
        # if output_type == "latent":
        #     video = latents
        # else:
        #     video_tensor = self.decode_latents(latents)
        #     video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        # if not return_dict:
        #     return video

        # return TextToVideoSDPipelineOutput(frames=video)
    
        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return TuneAVideoPipelineOutput(videos=video)