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

import torch
import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import imageio
from dataset.dataset_3D import Dataset_3D
from dataset.dataset_2D import Dataset_2D
from util import update_paths

def generate_sample_latent(args):
    cam_ids = args.cam_ids

    os.makedirs(args.latent_videos_dir,exist_ok=True)
    print(args.latent_videos_dir)

    for cam_id in cam_ids:
        args.cam_ids = [cam_id]
        if args.dataset == 'languagetable':
            dataset = Dataset_2D(args,mode=args.mode)
        else:
            dataset = Dataset_3D(args,mode=args.mode)
        batchsize = 32
        device = torch.device('cuda:0')
        data_loader = DataLoader(dataset=dataset, 
                                        batch_size=batchsize, 
                                        shuffle=False, 
                                        num_workers=32)
        for data in tqdm(data_loader,total=len(data_loader)):
            videos = data['latent']
            # acts = calculate_act_given_videos(video,batchsize,device,dims=2048)
            for video, episode_id, start_frame_id in zip(videos,data['video_name']['episode_id'],data['video_name']['start_frame_id']):
                video_name = episode_id+'_'+str(cam_id)+'_'+start_frame_id
                video_path = os.path.join(args.latent_videos_dir,video_name+'.pt')
                with open(video_path, 'wb') as file:
                    torch.save(video, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluation/languagetable/frame_ada.yaml")
    args = parser.parse_args()
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(data_config, config)
    config = OmegaConf.merge(diffusion_config, config)
    args = config
    

    args.debug = False
    args.pre_encode = True
    datasets = ['bridge','rt1','languagetable']
    modes = ['test','val']
    for mode in modes:
        for dataset in datasets:
            args.dataset = dataset
            args.mode = mode
            update_paths(args)
            generate_sample_latent(args)
