import os
import pandas as pd
from glob import glob
from omegaconf import OmegaConf
import random

from openai import OpenAI
import decord
import numpy as np
import torch
from PIL import Image

GUIDANCE_SCALE_V = 7.5
GUIDANCE_SCALE_I = 7.5

HEIGHT = 576
WIDTH = 320

DATA_PATH = "/home/qid/lutianyi/data"
TEMPLATE_CONFIG_PATH = '/home/qid/lutianyi/FLDM/configs/template_FLDM_zs_cnp2p.yaml'

video_output_path = '../data'
video_config_path = './data'
config_output_path = '../configs'
output_path = './outputs'

df = pd.read_csv(f"{DATA_PATH}/loveu-tgve-2024.csv")
sub_dfs = {
    # 'animal': df[0:41],
    # 'food': df[40:81],
    'scenery': df[112:121],
    # 'sport_activity': df[158:161],
    # 'vehicle': df[160:201],
}

# client = OpenAI(
#     api_key = "sk-dM6Ho34RUEMQW8Q2wIUTEFxbZqeOoArDDBVWzFmfeHsMrRId",
#     base_url = "https://api.fe8.cn/v1"
# )

client = OpenAI(
    api_key = "sk-yAA9L6BaNLhI8UqZFf1a509f9bF94b9bBf85Ff1826D6E4Bc",
    base_url = "http://rerverseapi.workergpt.cn/v1",
    timeout=1000
)

sample_rate = 4
n_frames = 16

def clip_frame(path, sample_rate):
    frames = decord.VideoReader(path)
    nframes = len(frames)
    num_frames = n_frames
    if sample_rate * (num_frames - 1) + 1 <= nframes:
        offset = random.randrange(nframes - (sample_rate * (num_frames - 1)))
        index = list(range(offset, nframes + offset, sample_rate))[:num_frames]
    else:
        index = list(
            np.linspace(0, nframes, num_frames, endpoint=False, dtype=int)
        )
    video = frames.get_batch(index)
    video = torch.from_numpy(video.asnumpy())
    video = video.permute((0, 3, 1, 2))
    return video




for sub_name, sub_df in sub_dfs.items():
    for index, row in sub_df.iterrows():
        config = OmegaConf.load(TEMPLATE_CONFIG_PATH)
        src_prompt = row['video caption']
        video_id = row['video id']
        cat = row['category']
        print(cat, video_id)

        # process prompts
        instructions = {x.lower(): str(row[x]).strip() for x in [
            "object_insertion",
            "object_removal",
            "object_change",
            "scene_change",
            "motion_change",
            "stylization"
        ]}
        # change instructions to prompts
        edit_prompts = {}
        for category, instruction in instructions.items():
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": "This is a caption of a video:" + src_prompt + "." + 
                            "If I want to edit the video according to this instruction:" + instruction + "." +
                            "please provide a new caption of the edited video.",
                    }
                ],
                model="gpt-4-1106-preview",
                # model="gpt-3.5-turbo",
            )
            edit_prompt = chat_completion.choices[0].message.content
            edit_prompts[category] = edit_prompt
        
        config.prompt_vs = edit_prompts
        config.prompt_is = instructions
        config.src_prompt = src_prompt

        # clip video and save frames
        id = str('%04d' % int(video_id))
        video_path = f"{DATA_PATH}/{cat}/{id}.mp4"
        # video = clip_frame(video_path, sample_rate)
        output_raw_vid_path = f"{video_output_path}/{cat}/{id}"
        # os.makedirs(output_raw_vid_path, exist_ok=True)
        # for idx in range(0, len(video)):
        #     frame = video[idx].permute(1, 2, 0)
        #     frame = Image.fromarray(frame.type(torch.uint8).numpy(),mode='RGB')
        #     frame_path = f"{output_raw_vid_path}/frame_{idx}.jpg"
        #     frame.save(frame_path)

        # build configs and save
        config.output_dir = f"{output_path}/{cat}/{video_id}"
        config.height = HEIGHT
        config.width = WIDTH
        config.video_path = f"{video_config_path}/{cat}/{id}"
        config.guidance_scale_v = GUIDANCE_SCALE_V
        config.guidance_scale_i = GUIDANCE_SCALE_I
        save_config_path = f"{config_output_path}/{cat}/{video_id}.yaml"
        os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
        OmegaConf.save(config, save_config_path)


            
        






# if __name__ == '__main__':
    