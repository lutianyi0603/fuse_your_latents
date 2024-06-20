import argparse
from omegaconf import OmegaConf
import torch
import os
from PIL import Image

from diffusers.models import UNet3DConditionModel, UNet2DConditionModel, ControlNetModel
from diffusers.utils.loading_utils import load_image
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from pipelines.pipeline_zs_cn import FusionEditPipeline
from tuneavideo.util import ddim_inversion, save_videos_grid

def main(args):
    config = OmegaConf.load(args.config)

    weight_dtype = torch.float16 if args.fp16 else torch.float32
    video_path = config.video_path
    height = config.height
    width = config.width
    height = 320
    width = 576
    num_frames = config.num_frames
    num_inv_steps = config.num_inv_steps
    num_infer_steps = config.num_inference_time
    output_dir = config.output_dir

    src_prompt = config.src_prompt
    prompt_vs = config.prompt_vs
    prompt_is = config.prompt_is

    denoise_time = config.denoise_time
    fusion_alpha_v = config.fusion_alpha_v
    fusion_alpha_i = config.fusion_alpha_i
    decay = config.decay
    reverse = config.reverse
    guidance_scale_v = config.guidance_scale_v
    guidance_scale_i = config.guidance_scale_i

    zs_model_path = config.pretrained_v_path
    cn_model_path = config.pretrained_i_path
    sd_path = config.pretrained_sd_path

    # load controlnet and sd_unet
    controlnet = ControlNetModel.from_pretrained(cn_model_path, torch_dtype=torch.float16)
    unet_2d = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet')
    # load zeroscope
    unet = UNet3DConditionModel.from_pretrained(zs_model_path, subfolder='unet')
    scheduler = DDIMScheduler.from_pretrained(zs_model_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(zs_model_path, subfolder='tokenizer')
    text_encoder_v = CLIPTextModel.from_pretrained(zs_model_path, subfolder='text_encoder')
    text_encoder_i = CLIPTextModel.from_pretrained(sd_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae')
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(zs_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(num_inv_steps)

    # pipeline
    pipe = FusionEditPipeline(
        vae=vae,
        text_encoder_v=text_encoder_v,
        text_encoder_i=text_encoder_i,
        tokenizer=tokenizer,
        unet_i=unet_2d,
        unet_v=unet,
        controlnet=controlnet,
        scheduler=scheduler
    ).to(weight_dtype).to('cuda')
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.schedule.config)
    # pipe.enable_model_cpu_offload()

    # load frames
    images = []
    files = os.listdir(video_path)
    for file in files:
        path = f"{video_path}/{file}"
        if not os.path.isdir(path):
            image = load_image(path)
            images.append(image.resize((width,height)))
    # images = [Image.open(x) for x in sorted(glob(f"{video_path}/*.jpg"))]

    ddim_inv_latent_v = ddim_inversion(
        pipe,
        ddim_scheduler=ddim_inv_scheduler,
        frames=images,
        num_inv_steps=num_inv_steps,
        prompt=src_prompt
    )[-1].to(weight_dtype)


    for edited_type, edited_prompt_i in prompt_is.items():
        save_path = f"{output_dir}/results/{edited_type}/{edited_prompt_i}.gif"
        edited_prompt_v = prompt_vs[edited_type]
        print(edited_prompt_i)
        print(edited_prompt_v)
        # if edited_type == 'stylization':
        #     fusion_alpha_v = 0
        #     fusion_alpha_i = 1
        #     denoise_time = 1
        video = pipe(
            prompt_i=edited_prompt_i,
            prompt_v=edited_prompt_v,
            image=images,
            fusion_alpha_v=fusion_alpha_v,
            fusion_alpha_i=fusion_alpha_i,
            height=height,
            width=width,
            latents_v=ddim_inv_latent_v,
            guidance_scale_v=guidance_scale_v,
            guidance_scale_i=guidance_scale_i,
            denoise_time=denoise_time,
            decay=decay,
            reverse=reverse,
            video_length=num_frames,
            num_inference_steps=num_inv_steps,
        ).videos

        save_videos_grid(video, save_path)
        print(f"Saved output to {save_path}")
    torch.cuda.empty_cache()


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--cfg_scale", type=float, default=12.5, help="classifier-free guidance scale")
    parser.add_argument("--fp16", action='store_true', help="use float16 for inference")
    args = parser.parse_args()

    main(args)