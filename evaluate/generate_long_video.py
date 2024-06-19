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

import os
import json
import torch
import imageio
import argparse
import shutil
import torchvision.transforms as T
from tqdm import tqdm
from einops import rearrange, repeat
from datetime import datetime
from diffusers.models import AutoencoderKL

from models import get_models
from dataset import get_dataset
from evaluate.generate_short_video import generate_single_video
from util import get_args, update_paths



def main(args,rank,thread,thread_num):
    device = torch.device("cuda", rank)
    args.latent_size = [t // 8 for t in args.video_size]
    model = get_models(args)
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)

    train_steps = int(args.evaluate_checkpoint.split('/')[-1][0:-3])
    current_date = datetime.now()
    experiment_dir = f"{args.results_dir}/{current_date.strftime('%m')}/{current_date.strftime('%d')}/{args.anno}-debug"
    episode_latent_videos_dir = f'{experiment_dir}/checkpoints/{train_steps:07d}/{args.mode}_episode_latent_videos'
    episode_videos_dir = f'{experiment_dir}/checkpoints/{train_steps:07d}/{args.mode}_episode_videos'

    os.makedirs(episode_latent_videos_dir, exist_ok=True)
    os.makedirs(episode_videos_dir, exist_ok=True)

    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:
            print('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                print('Ignoring: {}'.format(k))
        print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Successfully load model at {}!'.format(args.evaluate_checkpoint)) 
    model.to(device)
    model.eval()

    train_dataset,val_dataset = get_dataset(args)

    def printvideo(videos,filename):
        t_videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous()
        t_videos = rearrange(t_videos, 'f c h w -> f h w c')
        t_videos = t_videos.numpy()
        writer = imageio.get_writer(filename, fps=4) 
        for frame in t_videos:
            writer.append_data(frame) 
        writer.close()

    for sample_idx, ann_file in tqdm(enumerate(val_dataset.ann_files),total=len(val_dataset.ann_files)):
        if sample_idx % thread_num == thread:
            with open(ann_file, "rb") as f:
                ann = json.load(f)
            episode_id = ann['episode_id']
            ann_id = ann_file.split('/')[-1].split('.')[0]
            if args.dataset == 'languagetable':
                output_video_path = os.path.join(episode_videos_dir,f'{episode_id}.mp4')
                output_latents_path = os.path.join(episode_latent_videos_dir,f'{episode_id}.pt')
            else:
                output_video_path = os.path.join(episode_videos_dir,f'{ann_id}.mp4')
                output_latents_path = os.path.join(episode_latent_videos_dir,f'{ann_id}.pt')

            # if os.path.exists(output_latents_path) and os.path.exists(output_latents_path):
            #     continue
            
            if args.dataset == 'languagetable':
                video_path = os.path.join(args.video_path, ann['video_path'])
                latent_video_path = os.path.join(args.video_path, ann['latent_video_path'])
                with open(latent_video_path, 'rb') as file:
                    latent_video = torch.load(file)['obs']
            else:
                video_path = os.path.join(args.video_path, ann['videos'][0]['video_path'])
                latent_video_path = os.path.join(args.video_path, ann['latent_videos'][0]['latent_video_path'])
                with open(latent_video_path, 'rb') as file:
                    latent_video = torch.load(file)

            video_reader = imageio.get_reader(video_path)
            video_tensor = []
            for frame in video_reader:
                frame_tensor = torch.tensor(frame)
                video_tensor.append(frame_tensor)
            video_reader.close()
            video_tensor = torch.stack(video_tensor)

            if args.model == 'VDM':
                latent_video = video_tensor
                latent_video = latent_video.permute(0, 3, 1, 2)
                latent_video = val_dataset.preprocess(latent_video)

            total_frame = latent_video.size()[0]
            frame_ids = list(range(total_frame))
            if args.dataset == 'languagetable':
                action = torch.tensor(ann['actions'])
            else:
                arm_states, gripper_states = val_dataset._get_all_robot_states(ann, frame_ids)
                action = val_dataset._get_all_actions(arm_states, gripper_states, args.accumulate_action)

            action = action*val_dataset.c_act_scaler


            current_frame = 0
            start_image = latent_video[current_frame]
            seg_video_list = []
            seg_idx = 0
            
            latent_list = [latent_video[0:1].to(device)]

            while current_frame+args.num_frames-1 < total_frame:
                seg_action = action[current_frame:current_frame+args.num_frames-1]
                start_image = start_image.unsqueeze(0).unsqueeze(0)
                seg_action = seg_action.unsqueeze(0)
                seg_video, seg_latents = generate_single_video(args, start_image, seg_action, device, vae, model)
                seg_video = seg_video.squeeze()
                
                if args.model == 'VDM':
                    start_image = seg_video[-1].clone()
                else:
                    seg_latents = seg_latents.squeeze()
                    start_image = seg_latents[-1].clone()
                    latent_list.append(seg_latents[1:])

                current_frame += args.num_frames-1
                seg_video_list.append(seg_video[1:])
                seg_idx += 1
            
            seg_action = action[current_frame:]
            true_action = seg_action.size()[0]
            if true_action != 0:
                false_action = args.num_frames-true_action-1
                seg_false_action = repeat(seg_action[0], 'd -> f d', f= false_action) 
                
                com_action = torch.cat([seg_action,seg_false_action],dim=0)
                start_image = start_image.unsqueeze(0).unsqueeze(0)
                com_action = com_action.unsqueeze(0)
                seg_video, seg_latents = generate_single_video(args, start_image, com_action, device, vae, model)
                seg_video = seg_video.squeeze()
                seg_video_list.append(seg_video[1:true_action+1])
                if args.model != 'VDM':
                    seg_latents = seg_latents.squeeze()
                    latent_list.append(seg_latents[1:true_action+1])


            com_video = torch.cat(seg_video_list,dim=0).cpu()                 
            printvideo(com_video, output_video_path)
            print(output_video_path)

            if args.model != 'VDM':
                latents = torch.cat(latent_list,dim=0).cpu()    
                with open(output_latents_path,'wb') as file:
                    torch.save(latents,file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/rt1/frame_ada.yaml")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--thread", type=int, default=0)
    parser.add_argument("--thread-num", type=int, default=1)
    args = parser.parse_args()
    rank = args.rank
    thread = args.thread
    thread_num = args.thread_num
    args = get_args(args)
    update_paths(args)
    main(args,rank, thread, thread_num)