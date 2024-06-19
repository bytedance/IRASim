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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/evaluation/languagetable/frame_ada.yaml")
    args = parser.parse_args()
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(data_config, config)
    config = OmegaConf.merge(diffusion_config, config)
    args = config
    update_paths(args)

    args.debug = False
    args.pre_encode = False
    args.normalize = False
    act_list = []
    cam_ids = args.cam_ids

    videos_dir = f'opensource_robotdata/languagetable/evaluation_videos/{args.mode}_sample_videos'
    os.makedirs(videos_dir,exist_ok=True)
    def printvideo(videos,filename):
        # t_videos = rearrange(videos, 'f c h w -> f h w c')
        # t_videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
        t_videos = videos.detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
        # filename = f"{videos_dir}/train_steps_{train_steps}.mp4"
        # print(t_videos.shape)
        writer = imageio.get_writer(filename, fps=4) # fps 是帧率
        for idx,frame in enumerate(t_videos):
            writer.append_data(frame)

    for cam_id in cam_ids:
        args.cam_ids = [cam_id]
        # dataset = Dataset_3D(args,mode=mode)
        dataset = Dataset_2D(args,mode=args.mode)
        batchsize = 32
        device = torch.device('cuda:0')
        data_loader = DataLoader(dataset=dataset, 
                                        batch_size=batchsize, 
                                        shuffle=False, 
                                        num_workers=32)
        for data in tqdm(data_loader,total=len(data_loader)):
            videos = data['video'].permute(0,1,3,4,2)
            # acts = calculate_act_given_videos(video,batchsize,device,dims=2048)
            for video, episode_id, start_frame_id in zip(videos,data['video_name']['episode_id'],data['video_name']['start_frame_id']):
                video_name = episode_id+'_'+str(cam_id)+'_'+start_frame_id
                video_path = os.path.join(videos_dir,video_name+'.mp4')
                printvideo(video,video_path)