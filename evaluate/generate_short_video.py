# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imageio
import os
import argparse
import torch

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline
from dataset import get_dataset
from models import get_models
import torchvision.transforms.functional as F
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)



def generate_single_video(args, batch_start_frame, actions, device, vae, model):
    with torch.no_grad():
        mask_x = batch_start_frame.to(device).to(torch.float32)
        actions = actions.to(device).to(torch.float32)

        #################
        # denoise x
        #################
        if args.sample_method == 'PNDM':
            scheduler = PNDMScheduler.from_pretrained(args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, variance_type=args.variance_type)
        elif args.sample_method == 'DDPM':
            scheduler = DDPMScheduler.from_pretrained(args.scheduler_path, beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, variance_type=args.variance_type)

        videogen_pipeline = Trajectory2VideoGenPipeline(vae=vae, scheduler=scheduler, transformer=model)

        videos, latents = videogen_pipeline(
                                actions, 
                                mask_x = mask_x,
                                video_length=args.num_frames, 
                                height=args.video_size[0], 
                                width=args.video_size[1], 
                                num_inference_steps=args.infer_num_sampling_steps,
                                guidance_scale=args.guidance_scale, # dumpy
                                device = device,
                                return_dict = False,
                                output_type = 'both' if args.model != "VDM" else 'video'
                                )
        return videos, latents
        
def generate_sample_videos(args, val_dataloader, device, vae, ema, video_base_dir, frame_base_dir,latent_video_base_dir):
    os.makedirs(video_base_dir,exist_ok=True)
    os.makedirs(frame_base_dir,exist_ok=True)
    os.makedirs(latent_video_base_dir,exist_ok=True)
    for batch in tqdm(val_dataloader,total=len(val_dataloader),desc='predicting validation videos'):
        if args.model == 'VDM':
            x = batch['video'] 
        elif not args.pre_encode:
            video = batch['video'].to(device)
            b, f, _, _, _ = video.shape
            video = rearrange(video, 'b f c h w -> (b f) c h w').contiguous()
            stride = args.local_val_batch_size
            encode_video_list = []
            for i in range(0,video.size()[0],stride):
                encode_video_list.append(vae.encode(video[i:i+stride]).latent_dist.sample().mul_(vae.config.scaling_factor))
            encode_video = torch.cat(encode_video_list,dim=0)
            x = rearrange(encode_video, '(b f) c h w -> b f c h w',b=b,f=f)
        else:
            x = batch['latent']

        video_name = batch['video_name']
        batch_video_names_list = []
        for episode_id, cam_id, start_frame_id in zip(video_name['episode_id'],video_name['cam_id'],video_name['start_frame_id']):
            name = episode_id+'_' + cam_id + '_' + start_frame_id
            batch_video_names_list.append(name)

        batch_start_frame = x[:,0:1]
        actions = batch['action']
        pred_video_path_list = [os.path.join(video_base_dir,video_name+'.mp4')  for video_name in batch_video_names_list]
        pred_frame_dir_list = [os.path.join(frame_base_dir,video_name)  for video_name in batch_video_names_list]
        pred_latent_video_path_list = [os.path.join(latent_video_base_dir,video_name+'.pt')  for video_name in batch_video_names_list]
        if all(os.path.exists(path) for path in pred_video_path_list):
            print('skip')
            continue

        pred_videos, pred_latents = generate_single_video(args, batch_start_frame, actions, device, vae, ema)
        # video 1 16 3 256 320 latents 1 16 4 32 40
        b,_,_,_,_ = pred_videos.size()
        pred_videos = pred_videos.permute(0,1,3,4,2)
        pred_videos = ((pred_videos / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8)
        
        if pred_latents is None:
            if args.dataset == 'languagetable':
                pred_video_list = []
                t_pred_videos = rearrange(pred_videos,'b f h w c-> b f c h w')
                for video in t_pred_videos:
                    video = torch.stack([F.resize(frame, args.evaluate_video_size, antialias=True) for frame in video])
                    pred_video_list.append(video)
                pred_videos = torch.stack(pred_video_list)
                pred_videos = rearrange(pred_videos,'b f c h w -> b f h w c')
            t_pred_videos = (pred_videos/255.0 - 0.5 ) *2.0
            t_pred_videos = rearrange(t_pred_videos, 'b f h w c-> (b f) c h w')
            latents = process_frames_flexibly(t_pred_videos.float(), vae, device, batch_size=16)
            pred_latents = rearrange(latents,'(b f) c h w -> b f c h w', b=b)

        pred_videos = pred_videos.cpu().contiguous().numpy()

        for i, pred_latent_path in enumerate(pred_latent_video_path_list):
            with open(pred_latent_path,'wb') as file:
                torch.save(pred_latents[i].cpu(),file)

        for i, (pred_video_path,pred_frame_dir) in enumerate(zip(pred_video_path_list,pred_frame_dir_list)):
            pred_video = pred_videos[i]
            os.makedirs(pred_frame_dir,exist_ok=True)
            writer = imageio.get_writer(pred_video_path, fps=4) 
            for frame_idx, frame in enumerate(pred_video):
                writer.append_data(frame)
                if frame_idx != 0:
                    frame_file = os.path.join(pred_frame_dir, f"{frame_idx:06d}.png")
                    imageio.imwrite(frame_file, frame)
            writer.close()
    return

def process_frames_flexibly(frames, vae, device, batch_size=40): # 24 3 256 320
    with torch.no_grad():
        n_frames = len(frames)
        n_full_batches = n_frames // batch_size
        remainder = n_frames % batch_size
        
        processed_batches = []
        
        for i in range(n_full_batches):
            batch = frames[i*batch_size : (i+1)*batch_size]
            batch = batch.to(device)
            encoded = vae.encode(batch).latent_dist.sample()
            scaled = encoded.mul_(vae.config.scaling_factor).cpu()
            processed_batches.append(scaled)
        
        if remainder > 0:
            last_batch = frames[-remainder:] 
            last_batch = last_batch.to(device)
            encoded = vae.encode(last_batch).latent_dist.sample()
            scaled = encoded.mul_(vae.config.scaling_factor).cpu()
            processed_batches.append(scaled)
        
        processed_frames = torch.cat(processed_batches, dim=0)
        return processed_frames