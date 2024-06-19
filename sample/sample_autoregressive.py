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

from einops import rearrange, repeat
import imageio
import os
import argparse
from copy import deepcopy
from einops import rearrange
from models import get_models
from diffusion import create_diffusion,create_mask_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL,AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from dataset import get_dataset
from util import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, setup_distributed,
                   get_experiment_dir, text_preprocessing)
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
import numpy as np
import torch


def generate_single_video(args, batch, device, vae, mask_frame_num, model):
    with torch.no_grad():
        x = batch['start_image'].to(device).to(torch.float32)
        actions = batch['action'].to(device).to(torch.float32)
        f = actions.size()[0]+1

        # x = x.unsqueeze(0)
        # x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor) 

        mask_x = x.unsqueeze(0).unsqueeze(0)
        actions = actions.unsqueeze(0)

        #################
        # denoise x
        #################

        if args.sample_method == 'PNDM':
        if args.sample_method == 'PNDM':
            scheduler = PNDMScheduler.from_pretrained(args.scheduler_path, 
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule,
                                                variance_type=args.variance_type)
        elif args.sample_method == 'DDPM':
            scheduler = PNDMScheduler.from_pretrained(args.scheduler_path,
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule,
                                                variance_type=args.variance_type)
        else:
            assert False

        videogen_pipeline = Trajectory2VideoGenPipeline(vae=vae, 
                        #  text_encoder=text_encoder, 
                        #  tokenizer=tokenizer, 
                            scheduler=scheduler, 
                            transformer=model) # .to(device)

        # print('Generation total {} videos'.format(1))
        videos, latents = videogen_pipeline(
                                actions, 
                                mask_x = mask_x,
                                video_length=args.num_frames, 
                                height=args.video_size[0], 
                                width=args.video_size[1], 
                                num_inference_steps=args.infer_num_sampling_steps,
                                guidance_scale=args.guidance_scale, # dumpy
                                device = device,
                                return_dict = False
                                )
        # videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        return videos, latents

def process_frames_flexibly(frames, vae, batch_size=40):
    with torch.no_grad():
        n_frames = len(frames)
        # 计算需要多少个完整批次，以及最后一个批次需要包含的帧数
        n_full_batches = n_frames // batch_size
        remainder = n_frames % batch_size
        
        # 初始化列表来存储处理后的批次
        processed_batches = []
        
        # 处理完整的批次
        for i in range(n_full_batches):
            batch = frames[i*batch_size : (i+1)*batch_size]
            scaled = vae.decode(batch / vae.config.scaling_factor).sample
            processed_batches.append(scaled)
        
        # 如果有剩余的帧，处理这些帧
        if remainder > 0:
            last_batch = frames[-remainder:] # 获取最后 remainder 帧

            scaled = vae.decode(last_batch / vae.config.scaling_factor).sample

            processed_batches.append(scaled)
        
        # 合并所有处理后的批次
        processed_frames = torch.cat(processed_batches, dim=0)
        return processed_frames

def main(args,rank):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if os.path.exists('/mnt/bn/'):
        nas_dir = '/mnt/bn//'
    elif os.path.exists('/mnt/bn/'):
        nas_dir = '/mnt/bn//'
    else:
        nas_dir = '/mnt/bn//'
    args.nas_dir = nas_dir
    args.train_annotation_path = os.path.join(nas_dir+'/',args.train_annotation_path)
    args.val_annotation_path = os.path.join(nas_dir+'/',args.val_annotation_path)
    args.pretrained_model_path = os.path.join(nas_dir+'/',args.pretrained_model_path)
    args.vae_model_path = os.path.join(nas_dir+'/',args.vae_model_path)
    args.results_dir = os.path.join(nas_dir+'/',args.results_dir)

    device = torch.device("cuda", rank)
    # device = torch.device("cpu")

    # Create model:
    args.latent_size = [t // 8 for t in args.video_size]
    model = get_models(args)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if args.extras == 1:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    else:
        diffusion = create_mask_diffusion(timestep_respacing="")
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    if args.use_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_model_path, subfolder="t2v_required_models/vae_temporal_decoder").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)

    auto_video_path = os.path.join(os.path.dirname(os.path.dirname(args.pretrained)),f'auto_video_{args.anno}')
    os.makedirs(auto_video_path,exist_ok=True)
    print(auto_video_path)

    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            print('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print('Ignoring: {}'.format(k))
        print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Successfully load model at {}!'.format(args.pretrained)) 
    model.to(device)
    model.eval()
    # with open('batch.pickle', 'rb') as f:
    #     batch = pickle.load(f)
    mask_frame_num = args.mask_frame_num

    train_dataset,val_dataset = get_dataset(args)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=args.local_val_batch_size, shuffle=False, num_workers=args.num_workers)

    def printvideo(videos,filename):
        t_videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        t_videos = rearrange(t_videos, 'f c h w -> f h w c')
        t_videos = t_videos.numpy()
        # filename = f"{videos_dir}/train_steps_{train_steps}.mp4"
        # print(.format(1))
        print('Generate',t_videos.shape, 'shape video')
        writer = imageio.get_writer(filename, fps=4) 
        for frame in t_videos:
            writer.append_data(frame) # 1 4 13 23
    sample_idx = rank
    # for sample_idx in range(10):
        # if sample_idx < 8:
        #     continue
    ann_file = val_dataset.ann_files[sample_idx]
    with open(ann_file, "rb") as f:
        ann = torch.load(f)
    ori_video = ann['obs']
    action = ann['actions']
    action = action*val_dataset.c_act_scaler
    total_frame = ori_video.size()[0]
    # action_ids = ann['action_id']
    # ori_total_frame = len(action_ids)
    # # frame_ids = list(range(ori_total_frame))
    # frame_ids = []
    # for i in range(ori_total_frame):
    #     if action_ids[i] == -1:
    #         continue
    #     frame_ids.append(i)
    # total_frame = len(frame_ids)
    # ori_video = val_dataset._get_obs(ann, frame_ids)
    # action = val_dataset._get_actions(ann, frame_ids[0:-1])
    # action = torch.cat([action[0:1],action],dim=0)
    # video,actions

    current_frame = 0
    start_image = ori_video[current_frame]
    seg_video_list = []
    seg_idx = 0
    while current_frame+args.num_frames-1 < total_frame:
        seg_action = action[current_frame:current_frame+args.num_frames-1]
        batch = {'start_image':start_image,'action':seg_action}
        seg_video, seg_latents = generate_single_video(args, batch,device,vae,mask_frame_num,model)
        seg_video = seg_video.squeeze()
        seg_latents = seg_latents.squeeze()
        start_image = seg_latents[-1].clone()
        current_frame += args.num_frames-1
        seg_video_list.append(seg_video)
        printvideo(seg_video,f'{rank}-{seg_idx}.mp4')
        seg_idx += 1
    
    
    seg_action = action[current_frame:]
    true_action = seg_action.size()[0]
    if true_action != 0:
        false_action = args.num_frames-true_action-1
        seg_false_action = repeat(seg_action[0], 'd -> f d', f= false_action) 
        
        com_action = torch.cat([seg_action,seg_false_action],dim=0)
        batch = {'start_image':start_image,'action':com_action}
        seg_video, seg_latents = generate_single_video(args, batch,device,vae,mask_frame_num,model)
        seg_video = seg_video.squeeze()
        seg_video_list.append(seg_video[0:true_action+1])

    com_video_list = [seg_video_list[0]] + [video[1:] for video in seg_video_list[1:]]
    com_video = torch.cat(com_video_list,dim=0).cpu()
    

    ori_video = ori_video.to(device)
    ori_video = process_frames_flexibly(ori_video, vae, batch_size=20)
    ori_video = ori_video.cpu()


    com_video = torch.cat([ori_video,com_video],dim=2)
    
    printvideo(com_video,os.path.join(auto_video_path, f'{sample_idx}.mp4'))

    print()