# FLDM

This repository is the official implementation of **FLDM**.

[Fuse Your Latents: Video Editing with Multi-source Latent Diffusion Models](https://arxiv.org/abs/2310.16400)
Tianyi Lu, Xing Zhang, Jiaxi Gu, Renjing Pei, Songcen Xu, Xingjun Ma, Hang Xu, Zuxuan Wu


## Setup
```shell
pip install -r requirements.txt
```

## Models

* [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* [ControlNet + SD-1.5 using InstructPix-pix](https://huggingface.co/lllyasviel/control_v11e_sd15_ip2p)
* [zeroscope-v2-576w](https://huggingface.co/cerspense/zeroscope_v2_576w)

## Usage

```shell
./infer.sh
```

Note: This is the version of ZeroScope with ControlNet. You can load new models following:

```python
unet = UNet3DConditionModel.from_pretrained(model_path, subfolder='unet')
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder')
vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae')
```


